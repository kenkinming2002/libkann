#pragma once

#include <functional>
#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <random>
#include <assert.h>

namespace kann
{
  class ConvolutionalLayer : public Layer
  {
  public:
    LIBKANN_SYMEXPORT ConvolutionalLayer() = default;
    LIBKANN_SYMEXPORT ConvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount);

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const override;
    LIBKANN_SYMEXPORT size_t outputSize() const override;

  public:
    void randomize(std::default_random_engine& prng) override;

  public:
    LIBKANN_SYMEXPORT Eigen::VectorXd feedForward() override;
    LIBKANN_SYMEXPORT Eigen::RowVectorXd backPropagate(const Eigen::RowVectorXd& outputGradient) override;
    LIBKANN_SYMEXPORT void train(double learningRate) override;

  public:
    LIBKANN_SYMEXPORT std::unique_ptr<Layer> cross(const Layer& other, std::default_random_engine& engine, double mutationRate) const override;

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

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));

      archive(m_inputWidth);
      archive(m_inputHeight);
      archive(m_inputChannelCount);
      archive(m_outputChannelCount);

      archive(m_kernels);
      archive(m_kernelsGradient);
    }

  private:
    size_t m_inputWidth,  m_inputHeight;
    size_t m_kernelSize;
    size_t m_inputChannelCount;
    size_t m_outputChannelCount;

  private:
    std::vector<Eigen::MatrixXd> m_kernels;
    std::vector<Eigen::MatrixXd> m_kernelsGradient;
  };
}

CEREAL_REGISTER_TYPE(kann::ConvolutionalLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::ConvolutionalLayer);
