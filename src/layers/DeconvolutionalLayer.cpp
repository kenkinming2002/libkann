#include <libkann/layers/DeconvolutionalLayer.hpp>

#include <assert.h>

namespace kann
{
  DeconvolutionalLayer::DeconvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
    : m_inputWidth(inputWidth), m_inputHeight(inputHeight), m_kernelSize(kernelSize), m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount)
  {
    auto kernelsCount = m_inputChannelCount * m_outputChannelCount;
    m_kernels        .resize(kernelsCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
    m_kernelsGradient.resize(kernelsCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
  }

  size_t DeconvolutionalLayer::inputSize() const
  {
    return m_inputWidth * m_inputHeight * m_inputChannelCount;
  }

  size_t DeconvolutionalLayer::outputSize() const
  {
    return (m_inputWidth+m_kernelSize-1) * (m_inputHeight+m_kernelSize-1) * m_outputChannelCount;
  }

  void DeconvolutionalLayer::randomize(std::default_random_engine& engine)
  {
    std::uniform_real_distribution<double> dist(-0.1,0.1);
    for(auto& kernel : m_kernels)
      kernel = Eigen::MatrixXd::NullaryExpr(kernel.rows(), kernel.cols(), [&](){
        return dist(engine);
      });
  }

  Eigen::VectorXd DeconvolutionalLayer::feedForward()
  {
    // Well, we do deconvolutions
    Eigen::VectorXd output = Eigen::VectorXd::Zero(this->outputSize());

    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        auto outputWidth = m_inputWidth+m_kernelSize-1;
        auto outputHeight = m_inputHeight+m_kernelSize-1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel (this->input().data()  + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       outputChannel(output.data()   + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);
        for(size_t i=0; i<m_inputWidth; ++i)
          for(size_t j=0; j<m_inputHeight; ++j)
          {
            const auto& kernel = this->kernel(inputChannelIndex, outputChannelIndex);
            outputChannel.block(j,i, m_kernelSize, m_kernelSize) += inputChannel(j,i) * kernel;
          }
      }

    return output;
  }

  Eigen::RowVectorXd DeconvolutionalLayer::backPropagate(const Eigen::RowVectorXd& outputGradient)
  {
    assert(outputGradient.size() == this->outputSize());

    Eigen::RowVectorXd inputGradient = Eigen::VectorXd::Zero(this->inputSize());

    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        auto outputWidth = m_inputWidth+m_kernelSize-1;
        auto outputHeight = m_inputHeight+m_kernelSize-1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel         (this->input().data()        + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       inputGradientChannel (inputGradient.data()  + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);

        Eigen::Map<const Eigen::MatrixXd> outputGradientChannel(outputGradient.data() + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);
        for(size_t i=0; i<m_inputWidth; ++i)
          for(size_t j=0; j<m_inputHeight; ++j)
          {
            const auto& kernel   = this->kernel(inputChannelIndex, outputChannelIndex);
            auto& kernelGradient = this->kernelGradient(inputChannelIndex, outputChannelIndex);
            inputGradientChannel(j,i) += outputGradientChannel.block(j,i, m_kernelSize, m_kernelSize).cwiseProduct(kernel).sum();
            kernelGradient += outputGradientChannel.block(j,i, m_kernelSize, m_kernelSize) * inputChannel(j,i);
          }
      }

    return inputGradient;
  }

  void DeconvolutionalLayer::train(double learningRate)
  {
    for(size_t i=0; i<m_kernels.size(); ++i)
    {
      m_kernels[i] -= learningRate * m_kernelsGradient[i];
      m_kernelsGradient[i].setZero();
    }
  }

  std::unique_ptr<Layer> DeconvolutionalLayer::cross(const Layer& _other, std::default_random_engine& engine, double mutationRate) const
  {
    auto result = std::make_unique<DeconvolutionalLayer>(*this);
    const auto& other = dynamic_cast<const DeconvolutionalLayer&>(_other);

    std::uniform_int_distribution<> distribution(0, 1);
    std::uniform_real_distribution<double> mutationDistribution(0.0, 1.0);
    std::uniform_real_distribution<double> weightDistribution(-1.0, 1.0);

    assert(m_kernels.size() != 0);
    assert(m_kernels.size() == other.m_kernels.size());
    for(size_t channel = 0; channel<m_kernels.size(); ++channel)
    {
      auto& resultKernel = result->m_kernels[channel];
      const auto& kernel       = m_kernels[channel];
      const auto& otherKernel  = other.m_kernels[channel];

      assert(kernel.size() != 0);
      assert(kernel.size() == otherKernel.size());
      for(long i=0; i<resultKernel.size(); ++i)
      {
        resultKernel.data()[i] = distribution(engine) == 0 ? kernel.data()[i] : otherKernel.data()[i];
        if(mutationDistribution(engine) < mutationRate)
          resultKernel.data()[i] = weightDistribution(engine);
      }
    }

    return result;
  }
}
