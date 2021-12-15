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
  class DeconvolutionalLayer : public Layer
  {
  public:
    LIBKANN_SYMEXPORT DeconvolutionalLayer() = default;
    LIBKANN_SYMEXPORT DeconvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount);

  public:
    LIBKANN_SYMEXPORT std::unique_ptr<Layer> clone() const override;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const override;
    LIBKANN_SYMEXPORT size_t outputSize() const override;

  public:
    LIBKANN_SYMEXPORT void feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const override;
    LIBKANN_SYMEXPORT void backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const override;

  private:
    auto kernel(size_t inputChannelIndex, size_t outputChannelIndex) const
    {
      const size_t index = outputChannelIndex * m_inputChannelCount + inputChannelIndex;
      return Eigen::Map<const Eigen::MatrixXd>(params().data() + index * m_kernelSize * m_kernelSize, m_kernelSize, m_kernelSize);
    }

    auto kernelGradient(Eigen::ArrayXd& gradient, size_t inputChannelIndex, size_t outputChannelIndex) const
    {
      assert(gradient.size() == params().size());
      const size_t index = outputChannelIndex * m_inputChannelCount + inputChannelIndex;
      return Eigen::Map<Eigen::MatrixXd>(gradient.data() + index * m_kernelSize * m_kernelSize, m_kernelSize, m_kernelSize);
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
    }

  private:
    size_t m_inputWidth,  m_inputHeight;
    size_t m_kernelSize;
    size_t m_inputChannelCount;
    size_t m_outputChannelCount;
  };
}

CEREAL_REGISTER_TYPE(kann::DeconvolutionalLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::DeconvolutionalLayer);
