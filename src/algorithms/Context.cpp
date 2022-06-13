#include "Context.hpp"

#include <libkann/Tensor.hpp>
#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/Optimizer.hpp>

#include <libkann/operations/Subtract.hpp>
#include <libkann/operations/Scale.hpp>

namespace kann
{
  void Context::forward_pass(Layer& layer)
  {
    kann::LayerDef::Info info;
    input_index = graph.add_vertex();
    output_index = layer.def->process(graph, info, input_index);

    parameter_sizes      = std::move(info.parameter_sizes);
    parameter_tags       = std::move(info.parameter_tags);
    parameter_indices    = std::move(info.parameter_indices);

    state_sizes          = std::move(info.state_sizes);
    input_state_indices  = std::move(info.input_state_indices);
    output_state_indices = std::move(info.output_state_indices);
  }

  // TODO: Embed output size into graph vertex
  // TODO: Take in a loss function
  void Context::gradient_pass(size_t output_size)
  {
    expected_output_index = graph.add_vertex();

    // 1: Output Gradient
    size_t output_gradient_index = graph.add_vertex();
    size_t tmp_index             = graph.add_vertex();

    operation_t subtract_op = std::make_shared<SubtractOperation>(output_size);
    operation_t scale_op    = std::make_shared<ScaleOperation>(output_size, 2.0);

    graph.add_edge(std::move(subtract_op), {output_index, expected_output_index}, {tmp_index});
    graph.add_edge(std::move(scale_op),    {tmp_index}, {output_gradient_index});
    graph.set_gradient_index(output_index, output_gradient_index);

    // 2: State output gradient (set to zero)
    for(const auto& [size, index] : ranges::views::zip(state_sizes, output_state_indices))
    {
      size_t gradient_index = graph.add_vertex();
      state_gradient_indices.push_back(gradient_index);
      state_gradient_values.push_back(std::make_shared<const Tensor>(Tensor::constant(size, 0.0)));
      graph.set_gradient_index(index, gradient_index);
    }
  }

  void Context::backward_pass()
  {
    for(size_t parameter_index : parameter_indices)
    {
      size_t parameter_gradient_index = graph.get_gradient_index(parameter_index);
      parameter_gradient_indices.push_back(parameter_gradient_index);
    }
  }

  void Context::training_pass(const Optimizer& optimizer)
  {
    Optimizer::Info info;
    for(const auto& [size, parameter_index, parameter_gradient_index] : ranges::views::zip(parameter_sizes, parameter_indices, parameter_gradient_indices))
    {
      size_t new_parameter_index = optimizer.process(graph, info, size, parameter_index, parameter_gradient_index);
      new_parameter_indices.push_back(new_parameter_index);
    }

    optimizer_initial_state_values = std::move(info.initial_states);
    optimizer_input_state_indices  = std::move(info.input_states_indices);
    optimizer_output_state_indices = std::move(info.output_states_indices);
  }
}
