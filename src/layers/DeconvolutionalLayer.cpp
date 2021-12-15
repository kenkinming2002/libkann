#include <libkann/layers/DeconvolutionalLayer.hpp>

#include <assert.h>

namespace kann
{
  DeconvolutionalLayer::DeconvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
    : Layer(inputChannelCount * outputChannelCount * kernelSize * kernelSize),
      m_inputWidth(inputWidth), m_inputHeight(inputHeight),
      m_kernelSize(kernelSize),
      m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount) {}

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

  void DeconvolutionalLayer::feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const
  {
    // Well, we do deconvolutions
    output = Eigen::VectorXd::Zero(this->outputSize());

    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        auto outputWidth = m_inputWidth+m_kernelSize-1;
        auto outputHeight = m_inputHeight+m_kernelSize-1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel (input.data()  + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       outputChannel(output.data() + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);
        for(size_t i=0; i<m_inputWidth; ++i)
          for(size_t j=0; j<m_inputHeight; ++j)
          {
            auto kernel = this->kernel(inputChannelIndex, outputChannelIndex);
            outputChannel.block(j,i, m_kernelSize, m_kernelSize) += inputChannel(j,i) * kernel;
          }
      }
  }

  void DeconvolutionalLayer::backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const
  {
    assert(outputGradient.size() == this->outputSize());

    inputGradient = Eigen::VectorXd::Zero(this->inputSize());
    layerGradient.setZero(params().size());

    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        auto outputWidth = m_inputWidth+m_kernelSize-1;
        auto outputHeight = m_inputHeight+m_kernelSize-1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel         (input.data()        + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       inputGradientChannel (inputGradient.data()  + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);

        Eigen::Map<const Eigen::MatrixXd> outputGradientChannel(outputGradient.data() + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);
        for(size_t i=0; i<m_inputWidth; ++i)
          for(size_t j=0; j<m_inputHeight; ++j)
          {
            auto kernel         = this->kernel(inputChannelIndex, outputChannelIndex);
            auto kernelGradient = this->kernelGradient(layerGradient, inputChannelIndex, outputChannelIndex);
            inputGradientChannel(j,i) += outputGradientChannel.block(j,i, m_kernelSize, m_kernelSize).cwiseProduct(kernel).sum();
            kernelGradient += outputGradientChannel.block(j,i, m_kernelSize, m_kernelSize) * inputChannel(j,i);
          }
      }
  }
}
