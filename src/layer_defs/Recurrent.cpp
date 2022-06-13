#include <libkann/layer_defs/Recurrent.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Graph.hpp>
#include <libkann/Tensor.hpp>

#include <libkann/operations/Split2.hpp>
#include <libkann/operations/Concat2.hpp>

#include <range/v3/all.hpp>

#include <assert.h>

namespace kann
{
  YAML::Node RecurrentLayerDef::save(layer_def_t layer_def)
  {
    YAML::Node node;
    node["memory"] = static_pointer_cast<const RecurrentLayerDef>(layer_def)->m_memory_size;
    node["layers"] = layer_def->sub_layer_defs
      | ranges::views::transform([&](layer_def_t sub_layer_def) { return LayerDef::save(sub_layer_def); } )
      | ranges::to_vector;
    return node;
  }

  layer_def_t RecurrentLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<RecurrentLayerDef>();
    layer_def->m_memory_size = node["memory"].as<size_t>();

    auto layer_defs_node = node["layers"];
    layer_def->sub_layer_defs = layer_defs_node
      | ranges::views::transform([](YAML::Node child) { return LayerDef::load(child); })
      | ranges::to_vector;

    return layer_def;
  }

  RecurrentLayerDef::RecurrentLayerDef(size_t memory_size)
    : m_memory_size(memory_size) {}

  size_t RecurrentLayerDef::input_size() const
  {
    return sub_layer_defs.front()->input_size() - m_memory_size;
  }

  size_t RecurrentLayerDef::output_size() const
  {
    return sub_layer_defs.back()->output_size() - m_memory_size;
  }


  std::shared_ptr<Layer> RecurrentLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    layer->sub_layers = sub_layer_defs
      | ranges::views::transform([&prng](const auto& sub_layer_def) { return sub_layer_def->create(prng); })
      | ranges::to_vector;
    layer->states = { std::make_shared<const Tensor>(Tensor::constant(m_memory_size, 0.0)) };
    return layer;
  }

  size_t RecurrentLayerDef::process(Graph& graph, Info& info, size_t input_index) const
  {
    size_t real_input_index  = graph.add_vertex();
    size_t real_output_index = real_input_index;
    for(const layer_def_t& sub_layer_def : sub_layer_defs)
      real_output_index = sub_layer_def->process(graph, info, real_output_index);

    size_t memory_index      = graph.add_vertex();
    size_t new_memory_index  = graph.add_vertex();
    size_t output_index      = graph.add_vertex();

    operation_t concat2_op = std::make_shared<Concat2Operation>(this->input_size(),  m_memory_size);
    graph.add_edge(std::move(concat2_op), {input_index, memory_index}, {real_input_index});

    operation_t split2_op  = std::make_shared<Split2Operation>(this->output_size(), m_memory_size);
    graph.add_edge(std::move(split2_op), {real_output_index}, {output_index, new_memory_index});

    info.add_state(m_memory_size, memory_index, new_memory_index);
    return output_index;
  }
}

