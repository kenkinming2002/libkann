#pragma once

#include <libkann/LayerDef.hpp>

namespace kann
{
  class SequentialLayerDef : public LayerDef
  {
  public:
    SequentialLayerDef() = default;

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
    }
  };
}

CEREAL_REGISTER_TYPE(kann::SequentialLayerDef);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::LayerDef, kann::SequentialLayerDef)
