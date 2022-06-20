#include <libkann/layer_defs/Dense.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/LayerStorage.hpp>
#include <libkann/Graph.hpp>

#include <libkann/operations/TensorProduct.hpp>
#include <libkann/operations/Broadcast.hpp>
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

  std::shared_ptr<LayerStorage> DenseLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer_storage = std::make_shared<LayerStorage>();
    layer_storage->def = shared_from_this();
    layer_storage->parameters = {
      MutableTensor::normal(Shape::concat(m_input_shape, m_output_shape), prng, 0.0, 1.0 / std::sqrt(m_input_shape.size())).as_const(),
      MutableTensor::normal(m_output_shape,                               prng, 0.0, 1.0 / std::sqrt(m_input_shape.size())).as_const()
    };
    return layer_storage;
  }

  size_t DenseLayerDef::batch_process(Graph& graph, Info& info, size_t batch_size, size_t input_index) const
  {
    size_t output_index = graph.add_vertex();
    size_t weight_index = graph.add_vertex();
    size_t bias_index   = graph.add_vertex();

    size_t product_index = graph.add_vertex();
    size_t broadcast_index = graph.add_vertex();

    operation_t tensor_product_op = std::make_shared<TensorProductOperation>(1, m_output_shape.rank(), m_input_shape.rank());
    operation_t broadcast_op      = std::make_shared<BroadcastOperation>(Shape(batch_size));
    operation_t add_op            = std::make_shared<AddOperation>(Shape::concat(Shape(batch_size), m_output_shape));

    graph.add_edge(std::move(tensor_product_op), {input_index, weight_index},      {product_index});
    graph.add_edge(std::move(broadcast_op),      {bias_index},                     {broadcast_index});
    graph.add_edge(std::move(add_op),            {product_index, broadcast_index}, {output_index});

    info.add_parameter(Shape::concat(m_input_shape, m_output_shape),  tag, weight_index);
    info.add_parameter(m_output_shape,                                tag, bias_index);

    return output_index;
  }
}
