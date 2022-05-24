#pragma once

#include <libkann/layer_defs/Sequential.hpp>

namespace kann
{
  class RecurrentLayerDef : public SequentialLayerDef
  {
  public:
    RecurrentLayerDef() = default;
    RecurrentLayerDef(size_t memory_size);

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  protected:
    size_t states_count() const override;
    std::vector<size_t> states_sizes() const override;

  private:
    size_t m_memory_size;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<SequentialLayerDef>(this));
      archive(m_memory_size);
    }
  };
}

CEREAL_REGISTER_TYPE(kann::RecurrentLayerDef);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::SequentialLayerDef, kann::RecurrentLayerDef);

