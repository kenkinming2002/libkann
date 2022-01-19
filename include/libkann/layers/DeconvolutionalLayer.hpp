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

  private:
    LIBKANN_SYMEXPORT void initializeVariables();

  public:
    LIBKANN_SYMEXPORT std::unique_ptr<Layer> clone() const override;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const override;
    LIBKANN_SYMEXPORT size_t outputSize() const override;

  public:
    LIBKANN_SYMEXPORT std::vector<std::shared_ptr<const Variable>> parametersVariables() const override;
    LIBKANN_SYMEXPORT std::vector<std::shared_ptr<const Tensor>> parameters() const override;
    LIBKANN_SYMEXPORT void parameters(std::vector<std::shared_ptr<const Tensor>> parameters) override;

  public:
    LIBKANN_SYMEXPORT std::pair<std::shared_ptr<const Variable>, StateVariables> operator()(std::shared_ptr<const Variable> input, StateVariables state) const override;

  public:
    template<typename Archive>
    void save(Archive& archive) const
    {
      archive(cereal::base_class<Layer>(this));

      archive(m_inputWidth, m_inputHeight);
      archive(m_kernelSize);
      archive(m_inputChannelCount, m_outputChannelCount);

      archive(m_kernels);
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));

      archive(m_inputWidth, m_inputHeight);
      archive(m_kernelSize);
      archive(m_inputChannelCount, m_outputChannelCount);

      initializeVariables();

      archive(m_kernels);
    }

  private:
    size_t m_inputWidth,  m_inputHeight;
    size_t m_kernelSize;
    size_t m_inputChannelCount, m_outputChannelCount;

  private:
    std::vector<std::shared_ptr<const Variable>> m_kernelsVariable;
    std::vector<std::shared_ptr<const Tensor>>   m_kernels;
  };
}

CEREAL_REGISTER_TYPE(kann::DeconvolutionalLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::DeconvolutionalLayer);
