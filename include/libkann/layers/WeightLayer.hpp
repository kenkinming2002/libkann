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
    WeightLayer() = default;
    WeightLayer(size_t inputSize, size_t outputSize);

  public:
    WeightLayer(const WeightLayer& other);
    WeightLayer& operator=(const WeightLayer& other);

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
      archive(m_inputSize, m_outputSize);
      archive(m_weight, m_bias);
    }

  private:
    size_t m_inputSize, m_outputSize;

  private:
    std::shared_ptr<Parameter> m_weight;
    std::shared_ptr<Parameter> m_bias;
  };

}

CEREAL_REGISTER_TYPE(kann::WeightLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::WeightLayer);
