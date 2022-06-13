#pragma once

#include <libkann/LayerDef.hpp>

namespace kann
{
  class RecurrentLayerDef : public LayerDef
  {
  public:
    static YAML::Node save(layer_def_t layer_def);
    static layer_def_t load(YAML::Node node);

  public:
    RecurrentLayerDef() = default;
    RecurrentLayerDef(size_t memory_size);

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;
    size_t process(Graph& graph, Info& info, size_t input_index) const override;

  private:
    size_t m_memory_size;
  };
}

