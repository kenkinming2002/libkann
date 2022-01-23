#pragma once

#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/ActivationFunction.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <Eigen/Eigen>

namespace kann
{
  class ActivationLayer : public Layer
  {
  public:
    ActivationLayer() = default;
    ActivationLayer(size_t size, ActivationFunction activationFunction);

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
      archive(m_size);
      archive(m_activationFunction);
    }

  private:
    size_t m_size;
    ActivationFunction m_activationFunction;
  };
}

CEREAL_REGISTER_TYPE(kann::ActivationLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::ActivationLayer);
