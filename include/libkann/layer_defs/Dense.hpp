#pragma once

#include <libkann/LayerDef.hpp>

namespace kann
{
  class DenseLayerDef : public LayerDef
  {
  public:
    static YAML::Node save(layer_def_t layer_def);
    static layer_def_t load(YAML::Node node);

  public:
    DenseLayerDef() = default;

  public:
    Shape input_shape() const override;
    Shape output_shape() const override;

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;
    size_t process(Graph& graph, Info& info, size_t input_index) const override;

  private:
    Shape m_input_shape, m_output_shape;
  };
}
