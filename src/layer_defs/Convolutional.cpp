#include <libkann/layer_defs/Convolutional.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Graph.hpp>

#include <libkann/operations/ConvolutionOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>
#include <libkann/operations/CopyOperation.hpp>
#include <libkann/operations/SplitOperation.hpp>
#include <libkann/operations/ConcatOperation.hpp>

#include <range/v3/all.hpp>

#include <iostream>

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

  ConvolutionalLayerDef::ConvolutionalLayerDef(size_t input_channel_count, size_t output_channel_count, Vec2 input_size, Vec2 output_size, Vec2 kernel_size)
    : m_input_channel_count(input_channel_count), m_output_channel_count(output_channel_count),
      m_input_size(input_size), m_output_size(output_size), m_kernel_size(kernel_size) {}


  std::shared_ptr<Layer> ConvolutionalLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    layer->parameters = ranges::views::generate_n([&](){
      size_t kernel_size = m_kernel_size.width() * m_kernel_size.height();
      return std::make_shared<const Tensor>(Tensor::gaussian(kernel_size, prng, 0.0, 1.0 / kernel_size));
    }, m_input_channel_count * m_output_channel_count) | ranges::to_vector;
    return layer;
  }

  size_t ConvolutionalLayerDef::input_size() const
  {
    return m_input_size.width() * m_input_size.height() * m_input_channel_count;
  }

  size_t ConvolutionalLayerDef::output_size() const
  {
    return m_output_size.width() * m_output_size.height() * m_output_channel_count;
  }

  size_t ConvolutionalLayerDef::process(Graph& graph, Info& info, size_t input_index) const
  {
    auto output_index = graph.add_vertex();

    auto input_channel_indices  = graph.add_vertices(m_input_channel_count);
    auto output_channel_indices = graph.add_vertices(m_output_channel_count);

    auto pre_indices  = graph.add_vertices(m_input_channel_count * m_output_channel_count);
    auto post_indices = graph.add_vertices(m_input_channel_count * m_output_channel_count);

    auto kernel_indices = graph.add_vertices(m_input_channel_count * m_output_channel_count);

    // 1: Split/Concat
    operation_t split_op = std::make_shared<SplitOperation>(m_input_size.width() * m_input_size.height(), m_input_channel_count);
    graph.add_edge(std::move(split_op), {input_index}, input_channel_indices);

    operation_t concat_op = std::make_shared<ConcatOperation>(m_output_size.width() * m_output_size.height(), m_output_channel_count);
    graph.add_edge(std::move(concat_op), output_channel_indices, {output_index});

    // 2: Copy/Reduce
    for(const auto& [i, input_channel_index] : ranges::views::enumerate(input_channel_indices))
    {
      operation_t copy_op = std::make_shared<CopyOperation>(m_input_size.width() * m_input_size.height(), m_output_channel_count);
      std::vector<size_t> copy_outputs = ranges::views::ints(0uz, m_output_channel_count)
        | ranges::views::transform([&,i=i](size_t j) { return pre_indices[i * m_output_channel_count + j]; })
        | ranges::to_vector;
      graph.add_edge(std::move(copy_op), {input_channel_index}, std::move(copy_outputs));
    }

    for(const auto& [j, output_channel_index] : ranges::views::enumerate(output_channel_indices))
    {
      operation_t reduce_op = std::make_shared<ReduceOperation>(m_output_size.width() * m_output_size.height(), m_input_channel_count);
      std::vector<size_t> reduce_inputs = ranges::views::ints(0uz, m_input_channel_count)
        | ranges::views::transform([&,j=j](size_t i) { return post_indices[i * m_output_channel_count + j]; })
        | ranges::to_vector;
      graph.add_edge(std::move(reduce_op), std::move(reduce_inputs), {output_channel_index});
    }

    // 3: Convolution
    for(const auto& [kernel_index, pre_index, post_index] : ranges::views::zip(kernel_indices, pre_indices, post_indices))
    {
      // Don't ask me why we use cross correlation in a convolutional layer.
      operation_t op = std::make_shared<CrossCorrelationOperation>(m_input_size, m_output_size, m_kernel_size);
      graph.add_edge(std::move(op), {pre_index, kernel_index}, {post_index});
    }

    info.add_parameters(m_kernel_size.width() * m_kernel_size.height(), tag, kernel_indices);
    return output_index;
  }
}
