#pragma once

#include <libkann/LayerDef.hpp>
#include <libkann/ActivationFunction.hpp>

namespace kann
{
  class ActivationLayerDef : public LayerDef
  {
  public:
    ActivationLayerDef() = default;
    ActivationLayerDef(size_t size, ActivationFunction activationFunction);

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<LayerDef>(this));
      archive(m_size);
      archive(m_activationFunction);
    }

  private:
    size_t m_size;
    ActivationFunction m_activationFunction;
  };
}

CEREAL_REGISTER_TYPE(kann::ActivationLayerDef);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::LayerDef, kann::ActivationLayerDef);
