#include <libkann/layer_defs/Activation.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerStorage.hpp>

#include <libkann/Math.hpp>

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

  YAML::Node ActivationLayerDef::save(layer_def_t layer_def)
  {
    YAML::Node node;
    node["shape"]    = Shape::to_vector(std::static_pointer_cast<const ActivationLayerDef>(layer_def)->m_shape);
    node["function"] = to_string(std::static_pointer_cast<const ActivationLayerDef>(layer_def)->m_type);
    return node;
  }

  layer_def_t ActivationLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<ActivationLayerDef>();
    layer_def->m_shape = Shape::from_vector(node["shape"].as<std::vector<size_t>>());
    layer_def->m_type  = from_string(node["function"].as<std::string>());
    return layer_def;
  }

  Shape ActivationLayerDef::input_shape() const
  {
    return m_shape;
  }

  Shape ActivationLayerDef::output_shape() const
  {
    return m_shape;
  }

  std::shared_ptr<LayerStorage> ActivationLayerDef::create(std::default_random_engine& prng) const
  {
    return std::make_shared<LayerStorage>();
  }

  Tensor ActivationLayerDef::forward(Layer& layer, Tensor inputs) const
  {
    layer.saved_tensors = { inputs };
    return math::cwise(std::move(inputs), [this](float input)
    {
      switch(m_type)
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

  Tensor ActivationLayerDef::backward(Layer& layer, Tensor output_gradients) const
  {
    const Tensor& inputs = layer.saved_tensors[0];
    return math::cwise(inputs, output_gradients, [this](float input, float output_gradient)
    {
      float tmp;
      switch(m_type)
      {
      case ActivationLayerDef::Type::IDENTITY:
        return output_gradient;
      case ActivationLayerDef::Type::SIGMOID:
        tmp = std::exp(-input);
        return output_gradient * tmp / ((1+tmp) * (1+tmp));
      case ActivationLayerDef::Type::TANH:
        tmp = std::cosh(input);
        return output_gradient / (tmp * tmp);
      default:
        assert(false && "Unreachable");
      }
    });
  }
}
