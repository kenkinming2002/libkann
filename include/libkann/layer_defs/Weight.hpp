#pragma once

#include <libkann/LayerDef.hpp>

namespace kann
{
  class WeightLayerDef : public LayerDef
  {
  public:
    WeightLayerDef() = default;
    WeightLayerDef(size_t input_size, size_t output_size);

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  protected:
    size_t parameters_count() const override;
    std::vector<size_t> parameters_sizes() const override;

  private:
    size_t m_input_size, m_output_size;

  private:
    tensor_t m_weight;
    tensor_t m_bias;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<LayerDef>(this));
      archive(m_input_size, m_output_size);
      archive(m_weight, m_bias);
    }
  };
}

CEREAL_REGISTER_TYPE(kann::WeightLayerDef);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::LayerDef, kann::WeightLayerDef);
