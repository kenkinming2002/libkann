#include <libkann/layer_defs/Sequential.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerStorage.hpp>

#include <range/v3/all.hpp>

#include <assert.h>

namespace kann
{
  YAML::Node SequentialLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["layers"] = layer_def->sub_layer_defs
      | ranges::views::transform([&](std::shared_ptr<const LayerDef> sub_layer_def) { return LayerDef::save(sub_layer_def); } )
      | ranges::to_vector;
    return node;
  }

  std::shared_ptr<const LayerDef> SequentialLayerDef::load(YAML::Node node)
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

  tensor::Shape SequentialLayerDef::get_input_shape() const
  {
    return sub_layer_defs.front()->get_input_shape();
  }

  tensor::Shape SequentialLayerDef::get_output_shape() const
  {
    return sub_layer_defs.back()->get_output_shape();
  }

  tensor::Tensor<float> SequentialLayerDef::forward(Layer& layer, tensor::Tensor<float> inputs) const
  {
    tensor::Tensor<float> outputs = std::move(inputs);
    for(auto& sub_layer : layer.sub_layers)
      outputs = sub_layer->forward(std::move(outputs));

    return outputs;
  }

  tensor::Tensor<float> SequentialLayerDef::backward(Layer& layer, tensor::Tensor<float> output_gradients) const
  {
    tensor::Tensor<float> input_gradients = std::move(output_gradients);
    for(auto& sub_layer : layer.sub_layers | ranges::views::reverse)
      input_gradients = sub_layer->backward(std::move(input_gradients));

    return input_gradients;
  }
}
