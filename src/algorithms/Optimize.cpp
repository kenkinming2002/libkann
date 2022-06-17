#include <libkann/algorithms/Optimize.hpp>

#include "Context.hpp"

#include <libkann/Types.hpp>
#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/Executor.hpp>

#include <libkann/algorithms/ProgressBar.hpp>

#include <fstream>

namespace kann
{
  void optimize(Layer& layer, Tag tag, const Optimizer& optimizer, Executor& executor, const std::vector<Tensor>& inputs, const std::vector<Tensor>& expected_outputs)
  {
    Context context;
    context.forward_pass(layer);
    context.gradient_pass(layer.def->output_shape());
    context.backward_pass();
    context.training_pass(optimizer);

    {
      std::ofstream file("output/test2.dot");
      context.graph.write_graphviz(file);
    }

    // Executor
    Executor::Target target{
      .graph = std::move(context.graph),
      .input_indices  = {
        {context.input_index},
        {context.expected_output_index},
        std::move(context.parameter_indices),
        std::move(context.input_state_indices),
        std::move(context.optimizer_input_state_indices),
        std::move(context.state_gradient_indices)
      },
      .output_indices = {
        {context.output_index},
        std::move(context.new_parameter_indices),
        std::move(context.output_state_indices),
        std::move(context.optimizer_output_state_indices)
      }
    };

    // Compute
    std::vector<Tensor> parameters       = layer.get_parameters_all();
    std::vector<Tensor> states           = layer.get_states_all();

    std::vector<Tensor> state_gradient_values = std::move(context.state_gradient_values);
    std::vector<Tensor> optimizer_states      = std::move(context.optimizer_initial_state_values);

    std::vector<Tensor> outputs;
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
