#include "Context.hpp"

#include <libkann/Tensor.hpp>
#include <libkann/LayerStorage.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/Optimizer.hpp>

#include <libkann/operations/Subtract.hpp>
#include <libkann/operations/Scale.hpp>

namespace kann
{
  void Context::forward_pass(LayerStorage& layer, size_t batch_size)
  {
    kann::LayerDef::Info info;
    inputs_index = graph.add_vertex();
    outputs_index = layer.def->batch_process(graph, info, batch_size, inputs_index);

    this->batch_size = batch_size;

    parameter_shapes     = std::move(info.parameter_shapes);
    parameter_tags       = std::move(info.parameter_tags);
    parameter_indices    = std::move(info.parameter_indices);

    state_shapes         = std::move(info.state_shapes);
    input_state_indices  = std::move(info.input_state_indices);
    output_state_indices = std::move(info.output_state_indices);
  }

  // TODO: Embed output size into graph vertex
  // TODO: Take in a loss function
  void Context::gradient_pass(Shape output_shape)
  {
    Shape outputs_shape = Shape::concat(Shape(batch_size), output_shape);

    expected_outputs_index = graph.add_vertex();

    // 1: Output Gradient
    size_t output_gradient_index = graph.add_vertex();
    size_t tmp_index             = graph.add_vertex();

    operation_t subtract_op = std::make_shared<SubtractOperation>(outputs_shape);
    operation_t scale_op    = std::make_shared<ScaleOperation>(outputs_shape, 2.0);

    graph.add_edge(std::move(subtract_op), {outputs_index, expected_outputs_index}, {tmp_index});
    graph.add_edge(std::move(scale_op),    {tmp_index}, {output_gradient_index});
    graph.set_gradient_index(outputs_index, output_gradient_index);

    // 2: State output gradient (set to zero)
    for(const auto& [shape, index] : ranges::views::zip(state_shapes, output_state_indices))
    {
      size_t gradient_index = graph.add_vertex();
      state_gradient_indices.push_back(gradient_index);
      state_gradient_values.push_back(MutableTensor::constant(shape, 0.0).as_const());
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
    for(const auto& [shape, parameter_index, parameter_gradient_index] : ranges::views::zip(parameter_shapes, parameter_indices, parameter_gradient_indices))
    {
      size_t new_parameter_index = optimizer.process(graph, info, shape, parameter_index, parameter_gradient_index);
      new_parameter_indices.push_back(new_parameter_index);
    }

    optimizer_initial_state_values = std::move(info.initial_states);
    optimizer_input_state_indices  = std::move(info.input_states_indices);
    optimizer_output_state_indices = std::move(info.output_states_indices);
  }
}
