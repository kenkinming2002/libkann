#include <libkann/algorithms/Predict.hpp>

#include "Context.hpp"

#include <libkann/algorithms/ProgressBar.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Executor.hpp>

namespace kann
{
  std::vector<Tensor> predict(Layer& layer, Executor& executor, const std::vector<Tensor>& inputs)
  {
    Context context;
    context.forward_pass(layer);

    // Executor
    Executor::Target target{
      .graph = std::move(context.graph),
      .input_indices  = {
        {context.input_index},
        std::move(context.parameter_indices),
        std::move(context.input_state_indices)
      },
      .output_indices = {
        {context.output_index},
        std::move(context.output_state_indices)
      }
    };

    // Compute
    std::vector<Tensor> parameters = layer.get_parameters_all();
    std::vector<Tensor> states     = layer.get_states_all();

    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());

    ProgressBar progress_bar("training", inputs.size());
    for(const Tensor& input : inputs)
    {
      auto executor_inputs = {{input}, parameters, std::move(states)};
      auto executor_outputs = executor.run(target, std::move(executor_inputs));
      outputs.push_back(std::move(executor_outputs[0].front()));
      states = std::move(executor_outputs[1]);
      progress_bar.update("");
    }

    layer.set_states_all(std::move(states));

    return outputs;
  }
}
