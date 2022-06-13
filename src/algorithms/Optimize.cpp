#include <libkann/algorithms/Optimize.hpp>

#include <libkann/algorithms/ProgressBar.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/Graph.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Executor.hpp>
#include <libkann/Optimizer.hpp>

#include <libkann/operations/Subtract.hpp>
#include <libkann/operations/Scale.hpp>

#include <fstream>

namespace kann
{
  void optimize(Layer& layer, Tag tag, const Optimizer& optimizer, Executor& executor, const std::vector<tensor_t>& inputs, const std::vector<tensor_t>& expected_outputs)
  {
    kann::Graph graph;

    // Forward pass
    kann::LayerDef::Info layer_def_info;

    size_t input_index = graph.add_vertex();
    size_t output_index = layer.def->process(graph, layer_def_info, input_index);

    // Backward pass
    size_t expected_output_index = graph.add_vertex();
    size_t tmp_index             = graph.add_vertex();
    size_t output_gradient_index = graph.add_vertex();

    operation_t subtract_op = std::make_shared<SubtractOperation>(layer.def->output_size());
    operation_t scale_op    = std::make_shared<ScaleOperation>(layer.def->output_size(), 2.0);
    graph.add_edge(std::move(subtract_op), {output_index, expected_output_index}, {tmp_index});
    graph.add_edge(std::move(scale_op),    {tmp_index}, {output_gradient_index});
    graph.set_gradient_index(output_index, output_gradient_index);

    // Have zero gradient for state output
    std::vector<size_t> state_gradient_indices;
    std::vector<tensor_t> state_gradient_values;
    for(const auto& [size, index] : ranges::views::zip(layer_def_info.state_sizes, layer_def_info.output_state_indices))
    {
      size_t gradient_index = graph.add_vertex();
      state_gradient_indices.push_back(gradient_index);
      state_gradient_values.push_back(std::make_shared<const Tensor>(Tensor::constant(size, 0.0)));
      graph.set_gradient_index(index, gradient_index);
    }

    std::vector<size_t> new_paramter_indices;
    kann::Optimizer::Info optimizer_info;
    for(const auto& [size, index] : ranges::views::zip(layer_def_info.parameter_sizes, layer_def_info.parameter_indices))
    {
      // How do we know the size of the parameter
      size_t gradient_index = graph.get_gradient_index(index);
      size_t new_index = optimizer.process(graph, optimizer_info, size, index, gradient_index);
      new_paramter_indices.push_back(new_index);
    }

    {
      std::ofstream file("output/test2.dot");
      graph.write_graphviz(file);
    }

    // Executor
    Executor::Target target{
      .graph = std::move(graph),
      .input_indices  = {{input_index},  {expected_output_index}, std::move(layer_def_info.parameter_indices), std::move(layer_def_info.input_state_indices),  std::move(optimizer_info.input_states_indices), std::move(state_gradient_indices)},
      .output_indices = {{output_index},                          std::move(new_paramter_indices),             std::move(layer_def_info.output_state_indices), std::move(optimizer_info.output_states_indices)}
    };

    // Compute
    std::vector<tensor_t> parameters       = layer.get_parameters_all();
    std::vector<tensor_t> states           = layer.get_states_all();
    std::vector<tensor_t> optimizer_states = optimizer_info.initial_states;

    std::vector<tensor_t> outputs;
    outputs.reserve(inputs.size());

    ProgressBar progress_bar("training", inputs.size());
    for(const auto& [input, expected_output] : ranges::views::zip(inputs, expected_outputs))
    {
      auto executor_inputs = {{input}, {expected_output}, std::move(parameters), std::move(states), std::move(optimizer_states), state_gradient_values};
      auto executor_outputs = executor.run(target, std::move(executor_inputs));
      outputs.push_back(std::move(executor_outputs[0].front()));
      parameters       = std::move(executor_outputs[1]);
      states           = std::move(executor_outputs[2]);
      optimizer_states = std::move(executor_outputs[3]);
      progress_bar.update("");
    }

    layer.set_parameters_all(std::move(parameters));
    layer.set_states_all(std::move(states));
  }
}
