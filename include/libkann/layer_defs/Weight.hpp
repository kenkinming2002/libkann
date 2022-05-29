#pragma once

#include <libkann/LayerDef.hpp>

namespace kann
{
  class WeightLayerDef : public LayerDef
  {
  public:
    static YAML::Node save(layer_def_t layer_def);
    static layer_def_t load(YAML::Node node);

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
  };
}
