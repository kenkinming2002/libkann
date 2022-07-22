#include <libkann/layer_defs/Dense.hpp>

#include <libtensor/Tensor.hpp>

#include <libtensor/Initializer.hpp>
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

  std::shared_ptr<Layer> DenseLayerDef::create() const
  {
    auto layer = std::make_shared<DenseLayer>();
    layer->def = *this;
    return layer;
  }

  const LayerDef& DenseLayer::get_def() const { return def; }

  tensor::Shape DenseLayer::get_input_shape()  const { return def.input_shape; }
  tensor::Shape DenseLayer::get_output_shape() const { return def.output_shape; }

  void DenseLayer::initialize(std::default_random_engine& prng)
  {
    this->weight = Variable::create_normal(tensor::Shape::make(def.input_shape, def.output_shape), 0.0, 1.0 / std::sqrt(def.input_shape.size()), prng);
    this->bias   = Variable::create_normal(def.output_shape,                                       0.0, 1.0 / std::sqrt(def.input_shape.size()), prng);
  }

  std::unordered_map<std::string, const Variable*> DenseLayer::parameters_map() const { return {{"weight", &weight}, {"bias", &bias}}; }
  std::unordered_map<std::string, Variable*>       DenseLayer::parameters_map()       { return {{"weight", &weight}, {"bias", &bias}}; }

  tensor::Tensor<float> DenseLayer::forward(tensor::Tensor<float> inputs)
  {
    saved_tensors = { inputs };

    auto weight = this->weight.value;
    auto bias   = this->bias.value;

    inputs = inputs.flatten(tensor::Hint::single(),                      tensor::Hint::from_shape(get_input_shape()));
    weight = weight.flatten(tensor::Hint::from_shape(get_input_shape()), tensor::Hint::from_shape(get_output_shape()));
    bias   = bias  .flatten(tensor::Hint::from_shape(get_output_shape()));

    auto product = tensor::matrix_product<float>(inputs, false, weight, false);
    auto result = tensor::broadcast_add_outer<float>(product, bias);

    return result.unflatten(tensor::Hint::single(), tensor::Hint::from_shape(this->get_output_shape()));
  }

  tensor::Tensor<float> DenseLayer::backward(tensor::Tensor<float> output_gradients)
  {
    auto inputs = saved_tensors[0];
    auto weight = this->weight.value;
    auto bias   = this->bias.value;

    inputs = inputs.flatten(tensor::Hint::single(),                      tensor::Hint::from_shape(get_input_shape()));
    weight = weight.flatten(tensor::Hint::from_shape(get_input_shape()), tensor::Hint::from_shape(get_output_shape()));
    bias   = bias  .flatten(tensor::Hint::from_shape(get_output_shape()));

    output_gradients = output_gradients.flatten(tensor::Hint::single(), tensor::Hint::from_shape(get_output_shape()));

    auto input_gradients = tensor::matrix_product<float>(output_gradients, false, weight, true);
    auto weight_gradient = tensor::matrix_product<float>(inputs,  true, output_gradients, false);
    auto bias_gradient   = tensor::reduce_outer(output_gradients);

    input_gradients = input_gradients.unflatten(tensor::Hint::single(),                      tensor::Hint::from_shape(get_input_shape()));
    weight_gradient = weight_gradient.unflatten(tensor::Hint::from_shape(get_input_shape()), tensor::Hint::from_shape(get_output_shape()));
    bias_gradient   = bias_gradient  .unflatten(tensor::Hint::from_shape(get_output_shape()));

    this->weight.gradient = weight_gradient;
    this->bias.gradient   = bias_gradient;
    return input_gradients;
  }
}
