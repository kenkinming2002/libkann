#include <libkann/layers/DeconvolutionalLayer.hpp>

#include <assert.h>

namespace kann
{
  DeconvolutionalLayer::DeconvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
    : m_inputWidth(inputWidth), m_inputHeight(inputHeight),
      m_kernelSize(kernelSize),
      m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount)
  {
    m_kernels.resize(m_inputChannelCount * m_outputChannelCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
    m_kernelsGradient.resize(m_inputChannelCount * m_outputChannelCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
  }

  std::unique_ptr<Layer> DeconvolutionalLayer::clone() const
  {
    return std::make_unique<DeconvolutionalLayer>(*this);
  }

  size_t DeconvolutionalLayer::inputSize() const
  {
    return m_inputWidth * m_inputHeight * m_inputChannelCount;
  }

  size_t DeconvolutionalLayer::outputSize() const
  {
    return (m_inputWidth+m_kernelSize-1) * (m_inputHeight+m_kernelSize-1) * m_outputChannelCount;
  }

  Eigen::VectorXd DeconvolutionalLayer::feedForward()
  {
    Eigen::VectorXd output = Eigen::VectorXd::Zero(this->outputSize());

    // Well, we do deconvolutions
    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        auto outputWidth = m_inputWidth+m_kernelSize-1;
        auto outputHeight = m_inputHeight+m_kernelSize-1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel (input().data() + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       outputChannel(output.data()  + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);

        const auto& kernel = m_kernels[inputChannelIndex * m_outputChannelCount + outputChannelIndex];

        for(size_t i=0; i<m_inputWidth; ++i)
          for(size_t j=0; j<m_inputHeight; ++j)
            outputChannel.block(j,i, m_kernelSize, m_kernelSize) += inputChannel(j,i) * kernel;
      }

    return output;
  }

  Eigen::VectorXd DeconvolutionalLayer::backPropagate()
  {
    Eigen::VectorXd inputGradient = Eigen::VectorXd::Zero(this->inputSize());

    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        auto outputWidth = m_inputWidth+m_kernelSize-1;
        auto outputHeight = m_inputHeight+m_kernelSize-1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel         (input().data()          + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       inputGradientChannel (inputGradient.data()    + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<const Eigen::MatrixXd> outputGradientChannel(outputGradient().data() + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);

        const auto& kernel   = m_kernels[inputChannelIndex * m_outputChannelCount + outputChannelIndex];
        auto& kernelGradient = m_kernelsGradient[inputChannelIndex * m_outputChannelCount + outputChannelIndex];

        for(size_t i=0; i<m_inputWidth; ++i)
          for(size_t j=0; j<m_inputHeight; ++j)
          {
            inputGradientChannel(j,i) += outputGradientChannel.block(j,i, m_kernelSize, m_kernelSize).cwiseProduct(kernel).sum();
            kernelGradient += outputGradientChannel.block(j,i, m_kernelSize, m_kernelSize) * inputChannel(j,i);
          }
      }

    return inputGradient;
  }

  std::vector<std::span<double>> DeconvolutionalLayer::params()
  {
    std::vector<std::span<double>> params;
    std::transform(m_kernels.begin(), m_kernels.end(), std::back_inserter(params), [](Eigen::MatrixXd& kernel) {
      return std::span<double>(kernel.data(), kernel.size());
    });
    return params;
  }

  std::vector<std::span<const double>> DeconvolutionalLayer::params() const
  {
    std::vector<std::span<const double>> params;
    std::transform(m_kernels.begin(), m_kernels.end(), std::back_inserter(params), [](const Eigen::MatrixXd& kernel) {
      return std::span<const double>(kernel.data(), kernel.size());
    });
    return params;
  }

  std::vector<std::span<double>> DeconvolutionalLayer::paramsGradient()
  {
    std::vector<std::span<double>> paramsGradient;
    std::transform(m_kernelsGradient.begin(), m_kernelsGradient.end(), std::back_inserter(paramsGradient), [](Eigen::MatrixXd& kernelGradient) {
      return std::span<double>(kernelGradient.data(), kernelGradient.size());
    });
    return paramsGradient;
  }

  std::vector<std::span<const double>> DeconvolutionalLayer::paramsGradient() const
  {
    std::vector<std::span<const double>> paramsGradient;
    std::transform(m_kernelsGradient.begin(), m_kernelsGradient.end(), std::back_inserter(paramsGradient), [](const Eigen::MatrixXd& kernelGradient) {
      return std::span<const double>(kernelGradient.data(), kernelGradient.size());
    });
    return paramsGradient;
  }
}
