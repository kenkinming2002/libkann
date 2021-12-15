#include <libkann/layers/ConvolutionalLayer.hpp>

#include <assert.h>

namespace kann
{
  ConvolutionalLayer::ConvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
    : Layer(inputChannelCount * outputChannelCount * kernelSize * kernelSize),
      m_inputWidth(inputWidth), m_inputHeight(inputHeight),
      m_kernelSize(kernelSize),
      m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount)
  {
    /* Unsigned overflow is well-defined behavior so we could safely do the check
     * afterwards */
    assert(m_inputWidth  >= m_kernelSize);
    assert(m_inputHeight >= m_kernelSize);
  }

  std::unique_ptr<Layer> ConvolutionalLayer::clone() const
  {
    return std::make_unique<ConvolutionalLayer>(*this);
  }

  size_t ConvolutionalLayer::inputSize() const
  {
    return m_inputWidth * m_inputHeight * m_inputChannelCount;
  }

  size_t ConvolutionalLayer::outputSize() const
  {
    return (m_inputWidth-m_kernelSize+1) * (m_inputHeight-m_kernelSize+1) * m_outputChannelCount;
  }

  void ConvolutionalLayer::feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const
  {
    output = Eigen::VectorXd::Zero(this->outputSize());

    // Well, we do convolutions
    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        auto outputWidth = m_inputWidth-m_kernelSize+1;
        auto outputHeight = m_inputHeight-m_kernelSize+1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel (input.data()  + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       outputChannel(output.data() + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);
        for(size_t i=0; i<outputWidth; ++i)
          for(size_t j=0; j<outputHeight; ++j)
          {
            auto kernel = this->kernel(inputChannelIndex, outputChannelIndex);
            outputChannel(j,i) += inputChannel.block(j,i,m_kernelSize,m_kernelSize).cwiseProduct(kernel).sum();
          }
      }
  }

  void ConvolutionalLayer::backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const
  {
    assert(outputGradient.size() == this->outputSize());

    inputGradient = Eigen::VectorXd::Zero(this->inputSize());
    layerGradient.setZero(params().size());

    for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
      for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
      {
        auto outputWidth = m_inputWidth-m_kernelSize+1;
        auto outputHeight = m_inputHeight-m_kernelSize+1;

        Eigen::Map<const Eigen::MatrixXd> inputChannel         (input.data()          + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
        Eigen::Map<Eigen::MatrixXd>       inputGradientChannel (inputGradient.data()  + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);

        Eigen::Map<const Eigen::MatrixXd> outputGradientChannel(outputGradient.data() + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);
        for(size_t i=0; i<outputWidth; ++i)
          for(size_t j=0; j<outputHeight; ++j)
          {
            auto kernel         = this->kernel(inputChannelIndex, outputChannelIndex);
            auto kernelGradient = this->kernelGradient(layerGradient, inputChannelIndex, outputChannelIndex);
            inputGradientChannel.block(j,i,m_kernelSize,m_kernelSize) += outputGradientChannel(j,i) * kernel;
            kernelGradient                                            += outputGradientChannel(j,i) * inputChannel.block(j,i,m_kernelSize,m_kernelSize);
          }
      }
  }
}
