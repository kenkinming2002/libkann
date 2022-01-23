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
    DeconvolutionalLayer() = default;
    DeconvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount);

  public:
    DeconvolutionalLayer(const DeconvolutionalLayer& other);
    DeconvolutionalLayer& operator=(const DeconvolutionalLayer& other);

  public:
    std::unique_ptr<Layer> clone() const override;

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    std::vector<std::shared_ptr<const Parameter>> parameters(unsigned tags) const override;
    std::vector<std::shared_ptr<Parameter>> parameters(unsigned tags) override;

  public:
    std::pair<std::shared_ptr<const Variable>, StateVariables> operator()(std::shared_ptr<const Variable> input, StateVariables state) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));

      archive(m_inputWidth, m_inputHeight);
      archive(m_kernelSize);
      archive(m_inputChannelCount, m_outputChannelCount);

      archive(m_kernels);
    }

  private:
    size_t m_inputWidth,  m_inputHeight;
    size_t m_kernelSize;
    size_t m_inputChannelCount, m_outputChannelCount;

  private:
    std::vector<std::shared_ptr<Parameter>> m_kernels;
  };
}

CEREAL_REGISTER_TYPE(kann::DeconvolutionalLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::DeconvolutionalLayer);
