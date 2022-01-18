#pragma once

#include "libkann/Variable.hpp"
#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <Eigen/Eigen>

#include <random>
#include <assert.h>

namespace kann
{
  class WeightLayer : public Layer
  {
  public:
    LIBKANN_SYMEXPORT WeightLayer() = default;
    LIBKANN_SYMEXPORT WeightLayer(size_t inputSize, size_t outputSize);

  public:
    LIBKANN_SYMEXPORT WeightLayer(const WeightLayer& other);

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
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_inputSize, m_outputSize);
      archive(m_weight, m_bias);
    }

  private:
    size_t m_inputSize, m_outputSize;

  private:
    std::shared_ptr<const Variable> m_weightVariable = std::make_shared<const Variable>();
    std::shared_ptr<const Variable> m_biasVariable   = std::make_shared<const Variable>();
    std::shared_ptr<const Tensor> m_weight, m_bias;
  };

}

CEREAL_REGISTER_TYPE(kann::WeightLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::WeightLayer);
