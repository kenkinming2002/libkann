#include <libkann/layers/SoftMax.hpp>

#include <libkann/Layer.hpp>

#include <libtensor/Tensor.hpp>
#include <libtensor/Map.hpp>
#include <libtensor/Reduce.hpp>
#include <libtensor/Broadcast.hpp>

#include <fmt/core.h>

namespace kann
{
  YAML::Node SoftMaxLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["shape"] = tensor::Shape::to_vector(std::static_pointer_cast<const SoftMaxLayerDef>(layer_def)->shape);
    return node;
  }

  std::shared_ptr<const LayerDef> SoftMaxLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<SoftMaxLayerDef>();
    layer_def->shape = tensor::Shape::from_vector(node["shape"].as<std::vector<size_t>>());
    return layer_def;
  }

  std::shared_ptr<Layer> SoftMaxLayerDef::create() const
  {
    auto layer = std::make_shared<SoftMaxLayer>();
    layer->def = *this;
    return layer;
  }

  const LayerDef& SoftMaxLayer::get_def() const { return def; }

  tensor::Shape SoftMaxLayer::get_input_shape()  const { return def.shape; }
  tensor::Shape SoftMaxLayer::get_output_shape() const { return def.shape; }

  void SoftMaxLayer::initialize(std::default_random_engine& prng) {}

  std::unordered_map<std::string, const Variable*> SoftMaxLayer::parameters_map() const { return {}; }
  std::unordered_map<std::string, Variable*>       SoftMaxLayer::parameters_map()       { return {}; }

  tensor::Tensor<float> SoftMaxLayer::forward(tensor::Tensor<float> inputs)
  {
    inputs = inputs.flatten(tensor::Hint::single(), tensor::Hint::from_shape(get_input_shape()));

    auto exps    = tensor::unary_map(inputs, [](float input) { return std::exp(input); });
    auto factors = tensor::reduce_inner<float>(exps);
    auto outputs = tensor::broadcast_div_inner<float>(exps, factors);

    outputs = outputs.unflatten(tensor::Hint::single(), tensor::Hint::from_shape(get_output_shape()));
    saved_tensors = { outputs };
    return outputs;
  }

  tensor::Tensor<float> SoftMaxLayer::backward(tensor::Tensor<float> output_gradients)
  {
    auto outputs = saved_tensors[0];

    outputs          = outputs         .flatten(tensor::Hint::single(), tensor::Hint::from_shape(get_output_shape()));
    output_gradients = output_gradients.flatten(tensor::Hint::single(), tensor::Hint::from_shape(get_output_shape()));

    auto tmp1 = tensor::binary_map(output_gradients, outputs, [](auto a, auto b) { return a * b; });
    auto tmp2 = tensor::reduce_inner(tmp1);
    auto tmp3 = tensor::broadcast_mul_inner(outputs, tmp2);
    auto input_gradients = tensor::binary_map(tmp1, tmp3, [](auto a, auto b) { return a - b; });

    input_gradients = input_gradients.unflatten(tensor::Hint::single(), tensor::Hint::from_shape(get_input_shape()));

    return input_gradients;
  }
}

