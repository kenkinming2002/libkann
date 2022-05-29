#include <libkann/layer_defs/Recurrent.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Variable.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <range/v3/all.hpp>

#include <span>

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

  KANN_LAYER_DEF_SAVE_LOAD_REGISTER(recurrent, RecurrentLayerDef)

  RecurrentLayerDef::RecurrentLayerDef(size_t memory_size)
    : m_memory_size(memory_size) {}

  std::shared_ptr<Layer> RecurrentLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = SequentialLayerDef::create(prng);
    layer->states = { std::make_shared<const Tensor>(Tensor::constant(m_memory_size, 0.0)) };
    return layer;
  }

  size_t RecurrentLayerDef::input_size() const
  {
    return SequentialLayerDef::input_size() - m_memory_size;
  }

  size_t RecurrentLayerDef::output_size() const
  {
    return SequentialLayerDef::output_size() - m_memory_size;
  }

  static inline variable_t concat(variable_t variable1, variable_t variable2, size_t size1, size_t size2)
  {
    return Variable::apply(ReduceOperation(2), {
        Variable::apply(IdentityOperation(size1, size1 + size2, 0),     {variable1}),
        Variable::apply(IdentityOperation(size2, size1 + size2, size1), {variable2})
    });
  }

  static inline std::pair<variable_t, variable_t> split(variable_t variable, size_t size1, size_t size2)
  {
    return {
      Variable::apply(IdentityOperation(size1 + size2, size1, 0),     {variable}),
      Variable::apply(IdentityOperation(size1 + size2, size2, size1), {variable})
    };
  }

  LayerDef::ProcessOutput RecurrentLayerDef::process(ProcessInput process_input) const
  {
    // Remove memory state from input
    auto input        = process_input.variable;
    auto memory_input = process_input.states.front();

    process_input.states.erase(process_input.states.begin());
    process_input.variable = concat(input, memory_input, this->input_size(), m_memory_size);

    ProcessOutput process_output = SequentialLayerDef::process(std::move(process_input));
    auto [output, memory_output] = split(process_output.variable, this->output_size(), m_memory_size);

    process_output.states.insert(process_output.states.begin(), memory_output);
    process_output.variable = std::move(output);

    return process_output;
  }

  size_t RecurrentLayerDef::states_count() const
  {
    return 1;
  }

  std::vector<size_t> RecurrentLayerDef::states_sizes() const
  {
    return { m_memory_size };
  }
}

