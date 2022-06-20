#include <libkann/layer_defs/Activation.hpp>

#include <libkann/Graph.hpp>
#include <libkann/LayerStorage.hpp>

#include <libkann/Operation.hpp>
#include <libkann/operations/CWise.hpp>

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
    auto layer_storage = std::make_shared<LayerStorage>();
    layer_storage->def = shared_from_this();
    return layer_storage;
  }

  class ActivationGradientOperation : public Operation
  {
  public:
    constexpr ActivationGradientOperation(Shape shape, ActivationLayerDef::Type type)
      : m_shape(shape), m_type(type) {}

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override
    {
      return operation_process_cwise_impl<2,1>(std::move(inputs), m_shape, [this](float input, float output_gradient)
      {
        switch(m_type)
        {
          case ActivationLayerDef::Type::IDENTITY:
            return std::make_tuple(output_gradient);
          case ActivationLayerDef::Type::SIGMOID:
          {
            float tmp = std::exp(-input);
            return std::make_tuple(output_gradient * tmp / ((1+tmp) * (1+tmp)));
          }
          case ActivationLayerDef::Type::TANH:
          {
            float tmp = std::cosh(input);
            return std::make_tuple(output_gradient / (tmp * tmp));
          }
          default:
            assert(false && "Unreachable");
        }
      });
    }

  private:
    Shape m_shape;
    ActivationLayerDef::Type m_type;
  };

  class ActivationOperation : public Operation
  {
  public:
    constexpr ActivationOperation(Shape shape, ActivationLayerDef::Type type)
      : m_shape(shape), m_type(type) {}

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override
    {
      return operation_process_cwise_impl<1,1>(std::move(inputs), m_shape, [this](float input)
      {
        switch(m_type)
        {
        case ActivationLayerDef::Type::IDENTITY:
          return std::make_tuple(input);
        case ActivationLayerDef::Type::SIGMOID:
          return std::make_tuple(1.0f /  (1.0f + std::exp(-input)));
        case ActivationLayerDef::Type::TANH:
          return std::make_tuple(std::tanh(input));
        default:
          assert(false && "Unreachable");
        }
      });
    }

    operation_t differentiate() const override
    {
      return std::make_shared<ActivationGradientOperation>(m_shape, m_type);
    }

  private:
    Shape m_shape;
    ActivationLayerDef::Type m_type;
  };

  size_t ActivationLayerDef::batch_process(Graph& graph, Info& info, size_t batch_size, size_t input_index) const
  {
    size_t output_index = graph.add_vertex();
    operation_t op = std::make_shared<ActivationOperation>(Shape::concat(Shape(batch_size), m_shape), m_type);
    graph.add_edge(std::move(op), {input_index}, {output_index});
    return output_index;
  }
}
