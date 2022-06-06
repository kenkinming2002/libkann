#include <libkann/layer_defs/Activation.hpp>

#include <libkann/Graph.hpp>
#include <libkann/Layer.hpp>

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

  size_t ActivationLayerDef::input_size() const
  {
    return m_size;
  }

  size_t ActivationLayerDef::output_size() const
  {
    return m_size;
  }

  std::shared_ptr<Layer> ActivationLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    return layer;
  }

  class ActivationOperation : public CWiseOperation<ActivationOperation, 1, 1>
  {
  public:
    constexpr ActivationOperation(size_t size, ActivationLayerDef::Type type)
      : CWiseOperation<ActivationOperation, 1, 1>(size), m_type(type) {}

  public:
    cwise_outputs_t forward(cwise_inputs_t inputs) const
    {
      const auto& [input] = inputs;
      double output = [&,input=input]()
      {
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
      }();

      return {output};
    }

    cwise_inputs_t backward(cwise_inputs_t inputs, cwise_outputs_t output_gradients) const
    {
      const auto& [input] = inputs;
      const auto& [output_gradient] = output_gradients;

      double input_gradient = [&,input=input,output_gradient=output_gradient]()
      {
        switch(m_type)
        {
          case ActivationLayerDef::Type::IDENTITY:
            return output_gradient;
          case ActivationLayerDef::Type::SIGMOID:
          {
            double tmp = std::exp(-input);
            return output_gradient * tmp / ((1+tmp) * (1+tmp));
          }
          case ActivationLayerDef::Type::TANH:
          {
            double tmp = std::cosh(input);
            return output_gradient / (tmp * tmp);
          }
          default:
            assert(false && "Unreachable");
        }
      }();

      return {input_gradient};
    }

  private:
    ActivationLayerDef::Type m_type;
  };

  size_t ActivationLayerDef::process(Graph& graph, Info& info, size_t input_index) const
  {
    size_t output_index = graph.add_vertex();
    operation_t op = std::make_shared<ActivationOperation>(m_size, m_type);
    graph.add_edge(std::move(op), {input_index}, {output_index});
    return output_index;
  }
}
