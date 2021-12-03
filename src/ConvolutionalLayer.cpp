#include <libkann/ConvolutionalLayer.hpp>

#include <assert.h>

ConvolutionalLayer::ConvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
  : m_inputWidth(inputWidth), m_inputHeight(inputHeight), m_kernelSize(kernelSize), m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount)
{
  /* Unsigned overflow is well-defined behavior so we could safely do the check
   * afterwards */
  assert(m_inputWidth  >= m_kernelSize);
  assert(m_inputHeight >= m_kernelSize);

  m_input = Eigen::VectorXd::Zero(this->inputSize());

  auto kernelsCount = m_inputChannelCount * m_outputChannelCount;
  m_kernels        .resize(kernelsCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
  m_kernelsGradient.resize(kernelsCount, Eigen::MatrixXd::Zero(m_kernelSize, m_kernelSize));
}

Eigen::VectorXd ConvolutionalLayer::feedForward(Eigen::VectorXd input)
{
  assert(input.size() == this->inputSize());
  m_input = std::move(input);
  // Well, we do convolutions
  Eigen::VectorXd output = Eigen::VectorXd::Zero(this->outputSize());

  for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
    for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
    {
      auto outputWidth = m_inputWidth-m_kernelSize+1;
      auto outputHeight = m_inputHeight-m_kernelSize+1;

      Eigen::Map<const Eigen::MatrixXd> inputChannel (m_input.data()  + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
      Eigen::Map<Eigen::MatrixXd>       outputChannel(output.data()   + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);
      for(size_t i=0; i<outputWidth; ++i)
        for(size_t j=0; j<outputHeight; ++j)
        {
          const auto& kernel = this->kernel(inputChannelIndex, outputChannelIndex);
          outputChannel(j,i) += inputChannel.block(j,i,m_kernelSize,m_kernelSize).cwiseProduct(kernel).sum();
        }
    }

  return output;
}

Eigen::RowVectorXd ConvolutionalLayer::backPropagate(const Eigen::RowVectorXd& outputGradient)
{
  assert(outputGradient.size() == this->outputSize());

  Eigen::RowVectorXd inputGradient = Eigen::VectorXd::Zero(this->inputSize());

  for(size_t inputChannelIndex = 0; inputChannelIndex<m_inputChannelCount; ++inputChannelIndex)
    for(size_t outputChannelIndex = 0; outputChannelIndex<m_outputChannelCount; ++outputChannelIndex)
    {
      auto outputWidth = m_inputWidth-m_kernelSize+1;
      auto outputHeight = m_inputHeight-m_kernelSize+1;

      Eigen::Map<const Eigen::MatrixXd> inputChannel         (m_input.data()        + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);
      Eigen::Map<Eigen::MatrixXd>       inputGradientChannel (inputGradient.data()  + inputChannelIndex  * m_inputWidth * m_inputHeight, m_inputHeight, m_inputWidth);

      Eigen::Map<const Eigen::MatrixXd> outputGradientChannel(outputGradient.data() + outputChannelIndex * outputWidth  * outputHeight,  outputHeight,  outputWidth);
      for(size_t i=0; i<outputWidth; ++i)
        for(size_t j=0; j<outputHeight; ++j)
        {
          const auto& kernel   = this->kernel(inputChannelIndex, outputChannelIndex);
          auto& kernelGradient = this->kernelGradient(inputChannelIndex, outputChannelIndex);
          inputGradientChannel.block(j,i,m_kernelSize,m_kernelSize) += outputGradientChannel(j,i) * kernel;
          kernelGradient                                            += outputGradientChannel(j,i) * inputChannel.block(j,i,m_kernelSize,m_kernelSize);
        }
    }

  return inputGradient;
}

size_t ConvolutionalLayer::inputSize() const
{
  return m_inputWidth * m_inputHeight * m_inputChannelCount;
}

size_t ConvolutionalLayer::outputSize() const
{
  return (m_inputWidth-m_kernelSize+1) * (m_inputHeight-m_kernelSize+1) * m_outputChannelCount;
}


void ConvolutionalLayer::train(double learningRate)
{
  for(size_t i=0; i<m_kernels.size(); ++i)
  {
    m_kernels[i] -= learningRate * m_kernelsGradient[i];
    m_kernelsGradient[i].setZero();
  }
}

template<typename PRNG>
void ConvolutionalLayer::randomize(PRNG& prng)
{
  std::uniform_real_distribution<double> dist(-0.1,0.1);
  for(auto& kernel : m_kernels)
    kernel = Eigen::MatrixXd::NullaryExpr(kernel.rows(), kernel.cols(), [&](){
      return dist(prng);
    });
}

template void ConvolutionalLayer::randomize(std::mt19937&);
