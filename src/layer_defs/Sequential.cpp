#include <libkann/layer_defs/Sequential.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerStorage.hpp>

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

  std::shared_ptr<LayerStorage> SequentialLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer_storage = std::make_shared<LayerStorage>();
    layer_storage->sub_layer_storages = sub_layer_defs
      | ranges::views::transform([&prng](const auto& sub_layer_def) { return sub_layer_def->create(prng); })
      | ranges::to_vector;
    return layer_storage;
  }

  Shape SequentialLayerDef::input_shape() const
  {
    return sub_layer_defs.front()->input_shape();
  }

  Shape SequentialLayerDef::output_shape() const
  {
    return sub_layer_defs.back()->output_shape();
  }

  Tensor SequentialLayerDef::forward(Layer& layer, Tensor inputs) const
  {
    Tensor outputs = std::move(inputs);
    for(auto&& [sub_def, sub_storage] : ranges::views::zip(layer.def->sub_layer_defs, layer.storage->sub_layer_storages))
    {
      Layer layer;
      layer.def     = sub_def;
      layer.storage = sub_storage;
      outputs = layer.forward(std::move(outputs));
    }
    return outputs;
  }

  Tensor SequentialLayerDef::backward(Layer& layer, Tensor output_gradients) const
  {
    Tensor input_gradients = std::move(output_gradients);
    for(auto&& [sub_def, sub_storage] : ranges::views::zip(layer.def->sub_layer_defs, layer.storage->sub_layer_storages) | ranges::views::reverse)
    {
      Layer layer;
      layer.def     = sub_def;
      layer.storage = sub_storage;
      input_gradients = layer.forward(std::move(input_gradients));
    }
    return input_gradients;
  }
}
