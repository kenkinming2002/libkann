#pragma once

#include <libkann/export.hpp>
#include <libkann/Layer.hpp>

#include <Eigen/Eigen>

#include <random>
#include <assert.h>

class ConvolutionalLayer
{
public:
  ConvolutionalLayer() = default;

public:
  LIBKANN_SYMEXPORT ConvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount);

public:
  LIBKANN_SYMEXPORT Eigen::VectorXd feedForward(Eigen::VectorXd input);
  LIBKANN_SYMEXPORT Eigen::RowVectorXd backPropagate(const Eigen::RowVectorXd& outputGradient);

public:
  LIBKANN_SYMEXPORT size_t inputSize() const;
  LIBKANN_SYMEXPORT size_t outputSize() const;

public:
  LIBKANN_SYMEXPORT void train(double learningRate);

public:
  template<typename PRNG>
  void randomize(PRNG& prng);

private:
  Eigen::MatrixXd& kernel(size_t inputChannelIndex, size_t outputChannelIndex)
  {
    return m_kernels[outputChannelIndex * m_inputChannelCount + inputChannelIndex];
  }

  const Eigen::MatrixXd& kernel(size_t inputChannelIndex, size_t outputChannelIndex) const
  {
    return m_kernels[outputChannelIndex * m_inputChannelCount + inputChannelIndex];
  }

  Eigen::MatrixXd& kernelGradient(size_t inputChannelIndex, size_t outputChannelIndex)
  {
    return m_kernelsGradient[outputChannelIndex * m_inputChannelCount + inputChannelIndex];
  }

  const Eigen::MatrixXd& kernelGradient(size_t inputChannelIndex, size_t outputChannelIndex) const
  {
    return m_kernelsGradient[outputChannelIndex * m_inputChannelCount + inputChannelIndex];
  }

private:
  size_t m_inputWidth,  m_inputHeight;
  size_t m_kernelSize;
  size_t m_inputChannelCount;
  size_t m_outputChannelCount;

private:
  Eigen::VectorXd m_input;
  std::vector<Eigen::MatrixXd> m_kernels;
  std::vector<Eigen::MatrixXd> m_kernelsGradient;
};

static_assert(isLayer<ConvolutionalLayer>);
