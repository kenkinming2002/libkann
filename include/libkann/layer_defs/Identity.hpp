#pragma once

#include <libkann/LayerDef.hpp>

namespace kann
{
  class IdentityLayerDef : public LayerDef
  {
  public:
    IdentityLayerDef() = default;
    IdentityLayerDef(size_t inputSize, size_t outputSize, size_t offset);

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  private:
    size_t m_inputSize;
    size_t m_outputSize;
    size_t m_offset;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<LayerDef>(this));
      archive(m_inputSize);
      archive(m_outputSize);
      archive(m_offset);
    }
  };
}

CEREAL_REGISTER_TYPE(kann::IdentityLayerDef);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::LayerDef, kann::IdentityLayerDef);

