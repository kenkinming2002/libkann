#include <libkann/layer_defs/Dense.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Layer.hpp>
#include <libkann/Graph.hpp>

#include <libkann/operations/TensorProduct.hpp>
#include <libkann/operations/Add.hpp>

namespace kann
{
  YAML::Node DenseLayerDef::save(layer_def_t layer_def)
  {
    YAML::Node node;
    node["input_shape"]  = Shape::to_vector(std::static_pointer_cast<const DenseLayerDef>(layer_def)->m_input_shape);
    node["output_shape"] = Shape::to_vector(std::static_pointer_cast<const DenseLayerDef>(layer_def)->m_output_shape);
    return node;
  }

  layer_def_t DenseLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<DenseLayerDef>();
    layer_def->m_input_shape  = Shape::from_vector(node["input_shape"].as<std::vector<size_t>>());
    layer_def->m_output_shape = Shape::from_vector(node["output_shape"].as<std::vector<size_t>>());
    return layer_def;
  }

  Shape DenseLayerDef::input_shape() const
  {
    return m_input_shape;
  }

  Shape DenseLayerDef::output_shape() const
  {
    return m_output_shape;
  }

  std::shared_ptr<Layer> DenseLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    layer->parameters = {
      MutableTensor::normal(Shape::concat(m_output_shape, m_input_shape), prng, 0.0, 1.0 / std::sqrt(m_input_shape.size())).as_const(),
      MutableTensor::normal(m_output_shape,                               prng, 0.0, 1.0 / std::sqrt(m_input_shape.size())).as_const()
    };
    return layer;
  }

  size_t DenseLayerDef::process(Graph& graph, Info& info, size_t input_index) const
  {
    size_t output_index = graph.add_vertex();
    size_t weight_index = graph.add_vertex();
    size_t bias_index   = graph.add_vertex();
    size_t tmp_index    = graph.add_vertex();

    operation_t tensor_product_op = std::make_shared<TensorProductOperation>(m_output_shape.rank(), 0, m_input_shape.rank());
    operation_t add_op            = std::make_shared<AddOperation>(m_output_shape);

    graph.add_edge(std::move(tensor_product_op), {weight_index, input_index}, {tmp_index});
    graph.add_edge(std::move(add_op),            {tmp_index, bias_index},     {output_index});

    info.add_parameter(Shape::concat(m_output_shape, m_input_shape),  tag, weight_index);
    info.add_parameter(m_output_shape,                                tag, bias_index);

    return output_index;
  }
}
