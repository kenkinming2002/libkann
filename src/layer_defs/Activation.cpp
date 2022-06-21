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

  YAML::Node ActivationLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["shape"]    = Shape::to_vector(std::static_pointer_cast<const ActivationLayerDef>(layer_def)->m_shape);
    node["function"] = to_string(std::static_pointer_cast<const ActivationLayerDef>(layer_def)->m_type);
    return node;
  }

  std::shared_ptr<const LayerDef> ActivationLayerDef::load(YAML::Node node)
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

    MutableTensor outputs = MutableTensor::create(inputs.shape());
    math::transform(inputs.as_ref(), outputs.as_ref(), math::Operation::create([this](const float& input, float& output)
    {
      switch(m_type)
      {
      case ActivationLayerDef::Type::IDENTITY:
        output = input;
        break;
      case ActivationLayerDef::Type::SIGMOID:
        output = 1.0f /  (1.0f + std::exp(-input));
        break;
      case ActivationLayerDef::Type::TANH:
        output = std::tanh(input);
        break;
      default:
        assert(false && "Unreachable");
      }
    }));
    return outputs.as_const();
  }

  Tensor ActivationLayerDef::backward(Layer& layer, Tensor output_gradients) const
  {
    const Tensor& inputs = layer.saved_tensors[0];

    MutableTensor input_gradients = MutableTensor::create(inputs.shape());
    math::transform2(inputs.as_ref(), output_gradients.as_ref(), input_gradients.as_ref(), math::BinaryOperation::create([this](const float& input, const float& output_gradient, float& input_gradient)
    {
      float tmp;
      switch(m_type)
      {
      case ActivationLayerDef::Type::IDENTITY:
        input_gradient = output_gradient;
        break;
      case ActivationLayerDef::Type::SIGMOID:
        tmp = std::exp(-input);
        input_gradient = output_gradient * tmp / ((1+tmp) * (1+tmp));
        break;
      case ActivationLayerDef::Type::TANH:
        tmp = std::cosh(input);
        input_gradient = output_gradient / (tmp * tmp);
        break;
      default:
        assert(false && "Unreachable");
      }
    }));
    return input_gradients.as_const();
  }
}
