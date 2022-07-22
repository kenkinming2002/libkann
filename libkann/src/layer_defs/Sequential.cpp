#include <libkann/layer_defs/Sequential.hpp>

#include <libkann/Layer.hpp>

#include <range/v3/all.hpp>
#include <fmt/core.h>

#include <assert.h>

namespace kann
{
  YAML::Node SequentialLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["layers"] = std::static_pointer_cast<const SequentialLayerDef>(layer_def)->defs
      | ranges::views::transform([&](const auto& def) { return LayerDef::save(def); } )
      | ranges::to_vector;
    return node;
  }

  std::shared_ptr<const LayerDef> SequentialLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<SequentialLayerDef>();
    auto layer_defs_node = node["layers"];
    layer_def->defs = layer_defs_node
      | ranges::views::transform([](YAML::Node child) { return LayerDef::load(child); })
      | ranges::to_vector;

    return layer_def;
  }

  std::shared_ptr<Layer> SequentialLayerDef::create() const
  {
    auto layer = std::make_shared<SequentialLayer>();
    layer->def = *this;
    for(const auto& def : defs)
      layer->layers.push_back(def->create());
    return layer;
  }

  const LayerDef& SequentialLayer::get_def() const
  {
    return def;
  }

  tensor::Shape SequentialLayer::get_input_shape()  const { return layers.front()->get_input_shape(); }
  tensor::Shape SequentialLayer::get_output_shape() const { return layers.back()->get_output_shape(); }

  void SequentialLayer::initialize(std::default_random_engine& prng)
  {
    for(auto& layer : layers)
      layer->initialize(prng);
  }

  std::unordered_map<std::string, const Variable*> SequentialLayer::parameters_map() const
  {
    std::unordered_map<std::string, const Variable*> map;
    for(const auto& [i, layer] : ranges::views::enumerate(layers))
      for(const auto& [name, variable] : layer->parameters_map())
        map.insert({fmt::format("layer{}.{}", i, name), variable});

    return map;
  }

  std::unordered_map<std::string, Variable*> SequentialLayer::parameters_map()
  {
    std::unordered_map<std::string, Variable*> map;
    for(const auto& [i, layer] : ranges::views::enumerate(layers))
      for(const auto& [name, variable] : layer->parameters_map())
        map.insert({fmt::format("layer{}.{}", i, name), variable});

    return map;
  }

  tensor::Tensor<float> SequentialLayer::forward(tensor::Tensor<float> inputs)
  {
    auto outputs = std::move(inputs);
    for(auto& layer : layers)
      outputs = layer->forward(std::move(outputs));

    return outputs;
  }

  tensor::Tensor<float> SequentialLayer::backward(tensor::Tensor<float> output_gradients)
  {
    tensor::Tensor<float> input_gradients = std::move(output_gradients);
    for(auto& layer : ranges::views::reverse(layers))
      input_gradients = layer->backward(std::move(input_gradients));

    return input_gradients;
  }
}
