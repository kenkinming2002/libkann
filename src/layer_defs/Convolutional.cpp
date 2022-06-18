#include <libkann/layer_defs/Convolutional.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Layer.hpp>
#include <libkann/Graph.hpp>

#include <libkann/operations/CrossCorrelation.hpp>

namespace kann
{
  YAML::Node ConvolutionalLayerDef::save(layer_def_t layer_def)
  {
    YAML::Node node;
    node["input_channel_count"]  = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_input_channel_count;
    node["output_channel_count"] = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_output_channel_count;
    node["input_width"]          = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_input_size.width();
    node["input_height"]         = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_input_size.height();
    node["output_width"]         = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_output_size.width();
    node["output_height"]        = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_output_size.height();
    node["kernel_width"]         = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_kernel_size.width();
    node["kernel_height"]        = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_kernel_size.height();
    return node;
  }

  layer_def_t ConvolutionalLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<ConvolutionalLayerDef>();
    layer_def->m_input_channel_count  = node["input_channel_count"].as<size_t>();
    layer_def->m_output_channel_count = node["output_channel_count"].as<size_t>();
    layer_def->m_input_size           = Vec2(node["input_width"].as<size_t>(),  node["input_height"].as<size_t>());
    layer_def->m_output_size          = Vec2(node["output_width"].as<size_t>(), node["output_height"].as<size_t>());
    layer_def->m_kernel_size          = Vec2(node["kernel_width"].as<size_t>(), node["kernel_height"].as<size_t>());
    return layer_def;
  }

  std::shared_ptr<Layer> ConvolutionalLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    layer->parameters = { MutableTensor::normal(Shape{m_input_channel_count, m_output_channel_count, m_kernel_size.height(), m_kernel_size.width()},
      prng,
      0.0,
      1.0 / (m_kernel_size.width() * m_kernel_size.height())
    ).as_const() };
    return layer;
  }

  Shape ConvolutionalLayerDef::input_shape() const
  {
    return Shape{m_input_channel_count, m_input_size.height(), m_input_size.width()};
  }

  Shape ConvolutionalLayerDef::output_shape() const
  {
    return Shape{m_output_channel_count, m_output_size.height(), m_output_size.width()};
  }

  size_t ConvolutionalLayerDef::batch_process(Graph& graph, Info& info, size_t batch_size, size_t input_index) const
  {
    size_t output_index = graph.add_vertex();
    size_t kernels_index = graph.add_vertex();

    Vec2 padding_size = ((m_output_size - m_input_size) + (m_kernel_size - Vec2(1,1))) / 2;

    operation_t op = std::make_shared<CrossCorrelationOperation>(1, 1, 1, m_kernel_size, padding_size);
    graph.add_edge(std::move(op), {input_index, kernels_index}, {output_index});

    info.add_parameter(Shape{m_input_channel_count, m_output_channel_count, m_kernel_size.height(), m_kernel_size.width()}, tag, kernels_index);
    return output_index;
  }
}
