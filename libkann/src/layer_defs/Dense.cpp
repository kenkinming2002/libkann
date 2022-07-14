#include <libkann/layer_defs/Dense.hpp>

#include <libtensor/Tensor.hpp>

#include <libtensor/MatrixProduct.hpp>
#include <libtensor/Reduce.hpp>
#include <libtensor/Broadcast.hpp>

#include <libkann/Layer.hpp>

namespace kann
{
  YAML::Node DenseLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["input_shape"]  = tensor::Shape::to_vector(std::static_pointer_cast<const DenseLayerDef>(layer_def)->input_shape);
    node["output_shape"] = tensor::Shape::to_vector(std::static_pointer_cast<const DenseLayerDef>(layer_def)->output_shape);
    return node;
  }

  std::shared_ptr<const LayerDef> DenseLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<DenseLayerDef>();
    layer_def->input_shape  = tensor::Shape::from_vector(node["input_shape"].as<std::vector<size_t>>());
    layer_def->output_shape = tensor::Shape::from_vector(node["output_shape"].as<std::vector<size_t>>());
    return layer_def;
  }

  std::shared_ptr<LayerStorage> DenseLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer_storage = std::make_shared<LayerStorage>();

    Variable weight = Variable::create(tensor::Shape::concat(input_shape, output_shape));
    Variable bias   = Variable::create(output_shape);

    weight.value.fill_normal(prng, 0.0, 1.0 / std::sqrt(input_shape.size()));
    bias.value.fill_normal(prng, 0.0, 1.0 / std::sqrt(input_shape.size()));

    layer_storage->parameters.reserve(2);
    layer_storage->parameters.push_back(std::move(weight));
    layer_storage->parameters.push_back(std::move(bias));

    return layer_storage;
  }

  tensor::Shape DenseLayerDef::get_input_shape() const
  {
    return input_shape;
  }

  tensor::Shape DenseLayerDef::get_output_shape() const
  {
    return output_shape;
  }

  tensor::Tensor<const float> DenseLayerDef::forward(Layer& layer, tensor::Tensor<const float> inputs) const
  {
    layer.saved_tensors.clear();
    layer.saved_tensors.reserve(1);
    layer.saved_tensors.push_back(inputs);

    auto weight = layer.storage->parameters[0].value;
    auto bias   = layer.storage->parameters[1].value;

    inputs = inputs.flatten(tensor::Hint::single(),                      tensor::Hint::from_shape(get_input_shape()));
    weight = weight.flatten(tensor::Hint::from_shape(get_input_shape()), tensor::Hint::from_shape(get_output_shape()));
    bias   = bias  .flatten(tensor::Hint::from_shape(get_output_shape()));

    auto product = tensor::matrix_product<float>(inputs, false, weight, false);
    auto result = tensor::broadcast_add<tensor::Direction::LEFT, float>(product, bias);

    return result.unflatten(tensor::Hint::single(), tensor::Hint::from_shape(this->get_output_shape()));
  }

  tensor::Tensor<const float> DenseLayerDef::backward(Layer& layer, tensor::Tensor<const float> output_gradients) const
  {
    auto inputs = layer.saved_tensors[0];
    auto weight = layer.storage->parameters[0].value;
    auto bias   = layer.storage->parameters[1].value;

    inputs = inputs.flatten(tensor::Hint::single(),                      tensor::Hint::from_shape(get_input_shape()));
    weight = weight.flatten(tensor::Hint::from_shape(get_input_shape()), tensor::Hint::from_shape(get_output_shape()));
    bias   = bias  .flatten(tensor::Hint::from_shape(get_output_shape()));

    output_gradients = output_gradients.flatten(tensor::Hint::single(), tensor::Hint::from_shape(get_output_shape()));

    auto input_gradients = tensor::matrix_product<float>(output_gradients, false, weight, true);
    auto weight_gradient = tensor::matrix_product<float>(inputs,  true, output_gradients, false);
    auto bias_gradient   = tensor::reduce<tensor::Direction::LEFT>(output_gradients);

    input_gradients = input_gradients.unflatten(tensor::Hint::single(),                      tensor::Hint::from_shape(get_input_shape()));
    weight_gradient = weight_gradient.unflatten(tensor::Hint::from_shape(get_input_shape()), tensor::Hint::from_shape(get_output_shape()));
    bias_gradient   = bias_gradient  .unflatten(tensor::Hint::from_shape(get_output_shape()));

    layer.storage->parameters[0].gradient = weight_gradient;
    layer.storage->parameters[1].gradient = bias_gradient;
    return input_gradients;
  }
}
