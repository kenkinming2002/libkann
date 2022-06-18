#include <libkann/layer_defs/Sequential.hpp>

#include <libkann/Layer.hpp>

#include <range/v3/all.hpp>

#include <assert.h>

namespace kann
{
  YAML::Node SequentialLayerDef::save(layer_def_t layer_def)
  {
    YAML::Node node;
    node["layers"] = layer_def->sub_layer_defs
      | ranges::views::transform([&](layer_def_t sub_layer_def) { return LayerDef::save(sub_layer_def); } )
      | ranges::to_vector;
    return node;
  }

  layer_def_t SequentialLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<SequentialLayerDef>();
    auto layer_defs_node = node["layers"];
    layer_def->sub_layer_defs = layer_defs_node
      | ranges::views::transform([](YAML::Node child) { return LayerDef::load(child); })
      | ranges::to_vector;

    return layer_def;
  }

  Shape SequentialLayerDef::input_shape() const
  {
    return sub_layer_defs.front()->input_shape();
  }

  Shape SequentialLayerDef::output_shape() const
  {
    return sub_layer_defs.back()->output_shape();
  }


  std::shared_ptr<Layer> SequentialLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    layer->sub_layers = sub_layer_defs
      | ranges::views::transform([&prng](const auto& sub_layer_def) { return sub_layer_def->create(prng); })
      | ranges::to_vector;
    return layer;
  }

  size_t SequentialLayerDef::batch_process(Graph& graph, Info& info, size_t batch_size, size_t input_index) const
  {
    size_t output_index = input_index;
    for(const layer_def_t& sub_layer_def : sub_layer_defs)
      output_index = sub_layer_def->batch_process(graph, info, batch_size, output_index);

    return output_index;
  }
}
