#include <libkann/layer_defs/Activation.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Variable.hpp>

#include <libkann/operations/CWiseOperation.hpp>

namespace kann
{
  static inline std::string function_to_string(ActivationFunction function)
  {
    switch(function.type)
    {
    case ActivationFunction::Type::IDENTITY:
      return "identity";
    case ActivationFunction::Type::SIGMOID:
      return "sigmoid";
    case ActivationFunction::Type::TANH:
      return "tanh";
    default:
      throw std::runtime_error("unknown activation function type");
    }
  }

  static inline ActivationFunction string_to_function(std::string name)
  {
    if(name == "identity")
      return ActivationFunction(ActivationFunction::Type::IDENTITY);
    else if(name == "sigmoid")
      return ActivationFunction(ActivationFunction::Type::SIGMOID);
    else if(name == "tanh")
      return ActivationFunction(ActivationFunction::Type::TANH);
    else
      throw std::runtime_error("Unknown activation function type - " + name);
  }

  YAML::Node ActivationLayerDef::save(layer_def_t layer_def)
  {
    YAML::Node node;
    node["size"]     = std::static_pointer_cast<const ActivationLayerDef>(layer_def)->m_size;
    node["function"] = function_to_string(std::static_pointer_cast<const ActivationLayerDef>(layer_def)->m_activationFunction);
    return node;
  }

  layer_def_t ActivationLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<ActivationLayerDef>();
    layer_def->m_size               = node["size"].as<size_t>();
    layer_def->m_activationFunction = string_to_function(node["function"].as<std::string>());
    return layer_def;
  }

  ActivationLayerDef::ActivationLayerDef(size_t size, ActivationFunction activationFunction)
    : m_size(size), m_activationFunction(activationFunction) {}

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
    constexpr ActivationOperation(ActivationFunction activationFunction)
      : m_activationFunction(activationFunction) {}

  public:
    double forward(cwise_inputs_t inputs) const
    {
      const auto& [input] = inputs;
      return m_activationFunction.normal(input);
    }

    template<size_t index>
    double backward(double gradient, cwise_inputs_t inputs) const requires(index == 0)
    {
      const auto& [input] = inputs;
      return gradient * m_activationFunction.derivative(input);
    }

  private:
    ActivationFunction m_activationFunction;
  };

  LayerDef::ProcessOutput ActivationLayerDef::process(ProcessInput input) const
  {
    ProcessOutput output;
    output.variable = Variable::apply(ActivationOperation(m_activationFunction), {input.variable});
    output.states = input.states;
    return output;
  }
}
