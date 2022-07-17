#include <libkann/layer_defs/SoftMax.hpp>

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

  std::shared_ptr<LayerStorage> SoftMaxLayerDef::create(std::default_random_engine& prng) const
  {
    return std::make_shared<LayerStorage>();
  }

  tensor::Shape SoftMaxLayerDef::get_input_shape() const
  {
    return shape;
  }

  tensor::Shape SoftMaxLayerDef::get_output_shape() const
  {
    return shape;
  }

  tensor::Tensor<float> SoftMaxLayerDef::forward(Layer& layer, tensor::Tensor<float> inputs) const
  {
    inputs = inputs.flatten(tensor::Hint::single(), tensor::Hint::from_shape(get_input_shape()));

    auto exps    = tensor::unary_map(inputs, [](float input) { return std::exp(input); });
    auto factors = tensor::reduce_inner<float>(exps);
    auto outputs = tensor::broadcast_div_inner<float>(exps, factors);

    outputs = outputs.unflatten(tensor::Hint::single(), tensor::Hint::from_shape(get_output_shape()));

    // It is actually better to save outouts
    layer.saved_tensors.clear();
    layer.saved_tensors.reserve(1);
    layer.saved_tensors.push_back(outputs);

    return outputs;
  }

  tensor::Tensor<float> SoftMaxLayerDef::backward(Layer& layer, tensor::Tensor<float> output_gradients) const
  {
    auto outputs = layer.saved_tensors[0];

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

