#include <libkann/layer_defs/Activation.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Variable.hpp>

#include <libkann/operations/CWiseOperation.hpp>

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
    node["size"]     = std::static_pointer_cast<const ActivationLayerDef>(layer_def)->m_size;
    node["function"] = to_string(std::static_pointer_cast<const ActivationLayerDef>(layer_def)->m_type);
    return node;
  }

  layer_def_t ActivationLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<ActivationLayerDef>();
    layer_def->m_size = node["size"].as<size_t>();
    layer_def->m_type = from_string(node["function"].as<std::string>());
    return layer_def;
  }

  ActivationLayerDef::ActivationLayerDef(size_t size, Type type)
    : m_size(size), m_type(type) {}

  std::shared_ptr<Layer> ActivationLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    return layer;
  }

  size_t ActivationLayerDef::input_size() const
  {
    return m_size;
  }

  size_t ActivationLayerDef::output_size() const
  {
    return m_size;
  }

  class ActivationOperation : public CWiseOperation<ActivationOperation, 1>
  {
  public:
    constexpr ActivationOperation(ActivationLayerDef::Type type) : m_type(type) {}

  public:
    double forward(cwise_inputs_t inputs) const
    {
      const auto& [input] = inputs;
      switch(m_type)
      {
      case ActivationLayerDef::Type::IDENTITY:
        return input;
      case ActivationLayerDef::Type::SIGMOID:
        return 1.0 /  (1.0 + std::exp(-input));
      case ActivationLayerDef::Type::TANH:
        return std::tanh(input);
      default:
        assert(false && "Unreachable");
      }
    }

    template<size_t index>
    double backward(double gradient, cwise_inputs_t inputs) const requires(index == 0)
    {
      const auto& [input] = inputs;
      switch(m_type)
      {
        case ActivationLayerDef::Type::IDENTITY:
          return gradient;
        case ActivationLayerDef::Type::SIGMOID:
        {
          double tmp = std::exp(-input);
          return gradient * tmp / ((1+tmp) * (1+tmp));
        }
        case ActivationLayerDef::Type::TANH:
        {
          double tmp = std::cosh(input);
          return gradient / (tmp * tmp);
        }
        default:
          assert(false && "Unreachable");
      }
    }

  private:
    ActivationLayerDef::Type m_type;
  };

  LayerDef::ProcessOutput ActivationLayerDef::process(ProcessInput input) const
  {
    ProcessOutput output;
    output.variable = Variable::apply(ActivationOperation(m_type), {input.variable});
    output.states = input.states;
    return output;
  }
}
