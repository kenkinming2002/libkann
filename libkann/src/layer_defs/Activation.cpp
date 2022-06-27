#include <libkann/layer_defs/Activation.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerStorage.hpp>

#include <libtensor/Math.hpp>

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
    node["shape"]    = tensor::Shape::to_vector(std::static_pointer_cast<const ActivationLayerDef>(layer_def)->shape);
    node["function"] = to_string(std::static_pointer_cast<const ActivationLayerDef>(layer_def)->type);
    return node;
  }

  std::shared_ptr<const LayerDef> ActivationLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<ActivationLayerDef>();
    layer_def->shape = tensor::Shape::from_vector(node["shape"].as<std::vector<size_t>>());
    layer_def->type  = from_string(node["function"].as<std::string>());
    return layer_def;
  }

  tensor::Shape ActivationLayerDef::get_input_shape() const
  {
    return shape;
  }

  tensor::Shape ActivationLayerDef::get_output_shape() const
  {
    return shape;
  }

  std::shared_ptr<LayerStorage> ActivationLayerDef::create(std::default_random_engine& prng) const
  {
    return std::make_shared<LayerStorage>();
  }

  tensor::Tensor<const float> ActivationLayerDef::forward(Layer& layer, tensor::Tensor<const float> inputs) const
  {
    tensor::Tensor<float> outputs = tensor::Tensor<float>::create(inputs.shape());
    tensor::math::transform<1>(outputs.flatten(), {inputs.flatten()}, [this](float /*output*/, float input)
    {
      switch(type)
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

    layer.saved_tensors.clear();
    layer.saved_tensors.reserve(1);
    layer.saved_tensors.push_back(std::move(inputs));

    return outputs;
  }

  tensor::Tensor<const float> ActivationLayerDef::backward(Layer& layer, tensor::Tensor<const float> output_gradients) const
  {
    tensor::Tensor<const float> inputs = std::move(layer.saved_tensors[0]);

    tensor::Tensor<float> input_gradients = tensor::Tensor<float>::create(inputs.shape());
    tensor::math::transform<2>(input_gradients.flatten(), {inputs.flatten(), output_gradients.flatten()}, [this](float /*input_gradient*/, float input, float output_gradient)
    {
      float tmp;
      switch(type)
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
    return input_gradients;
  }
}
