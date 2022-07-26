#include <libkann/layers/Dense.hpp>

#include <libtensor/Tensor.hpp>

#include <libtensor/Initializer.hpp>
#include <libtensor/MatrixProduct.hpp>
#include <libtensor/Reduce.hpp>
#include <libtensor/Broadcast.hpp>

#include <libkann/SL.hpp>
#include <libkann/Layer.hpp>

namespace kann
{
  template<> YAML::Node save_layer_def_impl(const DenseLayerDef& def)
  {
    YAML::Node node;
    node["input_size"]  = def.input_size;
    node["output_size"] = def.output_size;
    return node;
  }

  template<> DenseLayerDef load_layer_def_impl(const YAML::Node& node)
  {
    DenseLayerDef def;
    def.input_size  = node["input_size"].as<size_t>();
    def.output_size = node["output_size"].as<size_t>();
    return def;
  }

  std::unique_ptr<Layer> DenseLayerDef::create() const
  {
    auto layer = std::make_unique<DenseLayer>();
    layer->def = *this;
    return layer;
  }

  const LayerDef& DenseLayer::get_def() const { return def; }

  tensor::Shape DenseLayer::get_input_shape()  const { return tensor::Shape::make(def.input_size); }
  tensor::Shape DenseLayer::get_output_shape() const { return tensor::Shape::make(def.output_size); }

  void DenseLayer::initialize(std::default_random_engine& prng)
  {
    this->weight = Variable::create_normal(tensor::Shape::make(def.input_size, def.output_size), 0.0, 1.0 / std::sqrt(def.input_size), prng);
    this->bias   = Variable::create_normal(tensor::Shape::make(def.output_size),                 0.0, 1.0 / std::sqrt(def.input_size), prng);
  }

  std::unordered_map<std::string, const Variable*> DenseLayer::parameters_map() const { return {{"weight", &weight}, {"bias", &bias}}; }
  std::unordered_map<std::string, Variable*>       DenseLayer::parameters_map()       { return {{"weight", &weight}, {"bias", &bias}}; }

  tensor::Tensor<float> DenseLayer::forward(tensor::Tensor<float> inputs)
  {
    saved_tensors = { inputs };
    auto weight = this->weight.value;
    auto bias   = this->bias.value;
    auto product = tensor::matrix_product<float>(inputs, false, weight, false);
    auto result = tensor::broadcast_add_outer<float>(product, bias);
    return result;
  }

  tensor::Tensor<float> DenseLayer::backward(tensor::Tensor<float> output_gradients)
  {
    auto inputs = saved_tensors[0];
    auto weight = this->weight.value;
    auto bias   = this->bias.value;
    auto input_gradients = tensor::matrix_product<float>(output_gradients, false, weight, true);
    auto weight_gradient = tensor::matrix_product<float>(inputs,  true, output_gradients, false);
    auto bias_gradient   = tensor::reduce_outer(output_gradients);
    this->weight.gradient = weight_gradient;
    this->bias.gradient   = bias_gradient;
    return input_gradients;
  }
}
