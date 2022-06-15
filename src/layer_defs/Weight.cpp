#include <libkann/layer_defs/Weight.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Layer.hpp>
#include <libkann/Graph.hpp>

#include <libkann/operations/MatrixProduct.hpp>
#include <libkann/operations/Add.hpp>

namespace kann
{
  YAML::Node WeightLayerDef::save(layer_def_t layer_def)
  {
    YAML::Node node;
    node["input_size"]  = std::static_pointer_cast<const WeightLayerDef>(layer_def)->m_input_size;
    node["output_size"] = std::static_pointer_cast<const WeightLayerDef>(layer_def)->m_output_size;
    return node;
  }

  layer_def_t WeightLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<WeightLayerDef>();
    layer_def->m_input_size  = node["input_size"].as<size_t>();
    layer_def->m_output_size = node["output_size"].as<size_t>();
    return layer_def;
  }

  WeightLayerDef::WeightLayerDef(size_t input_size, size_t output_size)
    : m_input_size(input_size), m_output_size(output_size) {}

  Shape WeightLayerDef::input_shape() const
  {
    return Shape{m_input_size};
  }

  Shape WeightLayerDef::output_shape() const
  {
    return Shape{m_output_size};
  }

  std::shared_ptr<Layer> WeightLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    layer->parameters = {
      MutableTensor::normal(Shape{m_output_size, m_input_size}, prng, 0.0, 1.0 / std::sqrt(m_input_size)).as_const(),
      MutableTensor::normal(Shape{m_output_size}              , prng, 0.0, 1.0 / std::sqrt(m_input_size)).as_const()
    };
    return layer;
  }

  size_t WeightLayerDef::process(Graph& graph, Info& info, size_t input_index) const
  {
    size_t output_index = graph.add_vertex();
    size_t weight_index = graph.add_vertex();
    size_t bias_index   = graph.add_vertex();
    size_t tmp_index    = graph.add_vertex();

    operation_t matrix_product_op = std::make_shared<MatrixProductOperation>();
    operation_t add_op            = std::make_shared<AddOperation>(Shape(m_output_size));

    graph.add_edge(std::move(matrix_product_op), {weight_index, input_index}, {tmp_index});
    graph.add_edge(std::move(add_op),            {tmp_index, bias_index},     {output_index});

    info.add_parameter(Shape{m_output_size, m_input_size}, tag, weight_index);
    info.add_parameter(Shape{m_output_size},               tag, bias_index);

    return output_index;
  }
}
