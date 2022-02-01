#pragma once

#include <libkann/NewLayer.hpp>
#include <libkann/ActivationFunction.hpp>

namespace kann
{
  class ActivationLayer : public NewLayer
  {
  public:
    ActivationLayer() = default;
    ActivationLayer(size_t size, ActivationFunction activationFunction);

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    LayerVariable operator()(Scope scope, LayerVariable input) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<NewLayer>(this));
      archive(m_size);
      archive(m_activationFunction);
    }

  private:
    size_t m_size;
    ActivationFunction m_activationFunction;
  };
}

CEREAL_REGISTER_TYPE(kann::ActivationLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::NewLayer, kann::ActivationLayer);
