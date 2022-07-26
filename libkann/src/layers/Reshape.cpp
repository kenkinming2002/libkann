#include <libkann/layers/Reshape.hpp>

#include <libkann/SL.hpp>
#include <libkann/Layer.hpp>

#include <libtensor/Map.hpp>

namespace kann
{
  template<> YAML::Node save_layer_def_impl(const ReshapeLayerDef& def)
  {
    YAML::Node node;
    node["input_shape"]    = tensor::Shape::to_vector(def.input_shape);
    node["output_shape"]    = tensor::Shape::to_vector(def.output_shape);
    return node;
  }

  template<> ReshapeLayerDef load_layer_def_impl(const YAML::Node& node)
  {
    ReshapeLayerDef def;
    def.input_shape = tensor::Shape::from_vector(node["input_shape"].as<std::vector<size_t>>());
    def.output_shape = tensor::Shape::from_vector(node["output_shape"].as<std::vector<size_t>>());
    return def;
  }

  std::unique_ptr<Layer> ReshapeLayerDef::create() const
  {
    auto layer = std::make_unique<ReshapeLayer>();
    layer->def = *this;
    return layer;
  }

  const LayerDef& ReshapeLayer::get_def() const { return def; }

  tensor::Shape ReshapeLayer::get_input_shape()  const { return def.input_shape; }
  tensor::Shape ReshapeLayer::get_output_shape() const { return def.output_shape; }

  void ReshapeLayer::initialize(std::default_random_engine& prng) {}

  std::unordered_map<std::string, const Variable*> ReshapeLayer::parameters_map() const { return {}; }
  std::unordered_map<std::string, Variable*>       ReshapeLayer::parameters_map()       { return {}; }

  tensor::Tensor<float> ReshapeLayer::forward(tensor::Tensor<float> inputs)
  {
    const size_t batch_size = inputs.shape.dimensions[0];
    return inputs.reshape(tensor::Shape::make(batch_size, def.output_shape));
  }

  tensor::Tensor<float> ReshapeLayer::backward(tensor::Tensor<float> output_gradients)
  {
    const size_t batch_size = output_gradients.shape.dimensions[0];
    return output_gradients.reshape(tensor::Shape::make(batch_size, def.input_shape));
  }
}
