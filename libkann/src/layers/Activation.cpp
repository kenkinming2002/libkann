#include <libkann/layers/Activation.hpp>

#include <libkann/Layer.hpp>

#include <libtensor/Map.hpp>

namespace kann
{
  static inline std::string to_string(ActivationLayerDef::Type type)
  {
    switch(type)
    {
    case ActivationLayerDef::Type::IDENTITY:
      return "identity";
    case ActivationLayerDef::Type::SIGMOID:
      return "sigmoid";
    case ActivationLayerDef::Type::TANH:
      return "tanh";
    default:
      throw std::runtime_error("unknown activation function type");
    }
  }

  static inline ActivationLayerDef::Type from_string(std::string name)
  {
    if(name == "identity")
      return ActivationLayerDef::Type::IDENTITY;
    else if(name == "sigmoid")
      return ActivationLayerDef::Type::SIGMOID;
    else if(name == "tanh")
      return ActivationLayerDef::Type::TANH;
    else
      throw std::runtime_error("Unknown activation function type - " + name);
  }

  template<> YAML::Node LayerDef::save_impl(const ActivationLayerDef& def)
  {
    YAML::Node node;
    node["shape"]    = tensor::Shape::to_vector(def.shape);
    node["function"] = to_string(def.type);
    return node;
  }

  template<> ActivationLayerDef LayerDef::load_impl(const YAML::Node& node)
  {
    ActivationLayerDef def;
    def.shape = tensor::Shape::from_vector(node["shape"].as<std::vector<size_t>>());
    def.type  = from_string(node["function"].as<std::string>());
    return def;
  }

  std::shared_ptr<Layer> ActivationLayerDef::create() const
  {
    auto layer = std::make_shared<ActivationLayer>();
    layer->def = *this;
    return layer;
  }

  const LayerDef& ActivationLayer::get_def() const { return def; }

  tensor::Shape ActivationLayer::get_input_shape()  const { return def.shape; }
  tensor::Shape ActivationLayer::get_output_shape() const { return def.shape; }

  void ActivationLayer::initialize(std::default_random_engine& prng) {}

  std::unordered_map<std::string, const Variable*> ActivationLayer::parameters_map() const { return {}; }
  std::unordered_map<std::string, Variable*>       ActivationLayer::parameters_map()       { return {}; }

  tensor::Tensor<float> ActivationLayer::forward(tensor::Tensor<float> inputs)
  {
    saved_tensors = {inputs};
    return tensor::unary_map(inputs, [this](float input)
    {
      switch(def.type)
      {
      case ActivationLayerDef::Type::IDENTITY:
        return input;
      case ActivationLayerDef::Type::SIGMOID:
        return 1.0f /  (1.0f + std::exp(-input));
      case ActivationLayerDef::Type::TANH:
        return std::tanh(input);
      default:
        assert(false && "Unreachable");
      }
    });
  }

  tensor::Tensor<float> ActivationLayer::backward(tensor::Tensor<float> output_gradients)
  {
    auto inputs = saved_tensors[0];
    return tensor::binary_map(inputs, output_gradients, [this](float input, float output_gradient)
    {
      float tmp;
      switch(def.type)
      {
      case ActivationLayerDef::Type::IDENTITY:
        return output_gradient;
      case ActivationLayerDef::Type::SIGMOID:
        tmp = std::exp(-input);
        return output_gradient / ( (1.0f + tmp) * (1.0f + 1.0f / tmp) );
      case ActivationLayerDef::Type::TANH:
        tmp = std::cosh(input);
        return output_gradient / (tmp * tmp);
      default:
        assert(false && "Unreachable");
      }
    });
  }
}
