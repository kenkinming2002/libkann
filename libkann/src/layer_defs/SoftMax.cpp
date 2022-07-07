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

  tensor::Tensor<const float> SoftMaxLayerDef::forward(Layer& layer, tensor::Tensor<const float> inputs) const
  {
    inputs = inputs.flatten(tensor::flatten_single, this->get_input_shape());

    auto exps    = tensor::unary_map(inputs, [](float input) { return std::exp(input); });
    auto factors = tensor::reduce<tensor::Direction::RIGHT, float>(exps);
    auto outputs = tensor::broadcast_div<tensor::Direction::RIGHT, float>(exps, factors);

    outputs = outputs.unflatten(tensor::flatten_single, this->get_output_shape());

    // It is actually better to save outouts
    layer.saved_tensors.clear();
    layer.saved_tensors.reserve(1);
    layer.saved_tensors.push_back(outputs);

    return outputs;
  }

  tensor::Tensor<const float> SoftMaxLayerDef::backward(Layer& layer, tensor::Tensor<const float> output_gradients) const
  {
    auto outputs = layer.saved_tensors[0];
    auto input_gradients = tensor::Tensor<float>::create(output_gradients.shape());

    outputs          = outputs         .flatten(tensor::flatten_single, this->get_output_shape());
    output_gradients = output_gradients.flatten(tensor::flatten_single, this->get_output_shape());
    input_gradients  = input_gradients .flatten(tensor::flatten_single, this->get_input_shape());

    const size_t batch_size = outputs.dimension(0), size = outputs.dimension(1);
    input_gradients.fill(0.0f);
    for(size_t k=0; k<batch_size; ++k)
      for(size_t i=0; i<size; ++i)
        for(size_t j=0; j<size; ++j)
        {
          if(i == j)
            input_gradients(k, i) += output_gradients(k, j) * outputs(k, j) * (1 - outputs(k, i));
          else
            input_gradients(k, i) -= output_gradients(k, j) * outputs(k, j) * outputs(k, i);
        }

    input_gradients = input_gradients.unflatten(tensor::flatten_single, this->get_input_shape());
    return input_gradients;
  }
}

