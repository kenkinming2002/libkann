#include <libkann/algorithms/Predict.hpp>

#include "Context.hpp"
#include "Batch.hpp"

#include <libkann/algorithms/ProgressBar.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Executor.hpp>

namespace kann
{
  std::vector<Tensor> predict(Layer& layer, Executor& executor, const std::vector<Tensor>& inputs)
  {
    Context context;
    context.forward_pass(layer, 1);

    // Executor
    Executor::Target target{
      .graph = std::move(context.graph),
      .input_indices  = {
        {context.inputs_index},
        std::move(context.parameter_indices),
        std::move(context.input_state_indices)
      },
      .output_indices = {
        {context.outputs_index},
        std::move(context.output_state_indices)
      }
    };

    // Compute
    std::vector<Tensor> parameters = layer.get_parameters_all();
    std::vector<Tensor> states     = layer.get_states_all();

    std::vector<Tensor> input_batches = batch(inputs, 1);

    std::vector<Tensor> output_batches;
    output_batches.reserve(input_batches.size());
    ProgressBar progress_bar("testing", inputs.size());
    for(const Tensor& input_batch : input_batches)
    {
      auto executor_inputs = {{input_batch}, parameters, std::move(states)};
      auto executor_outputs = executor.run(target, std::move(executor_inputs));
      output_batches.push_back(std::move(executor_outputs[0].front()));
      states = std::move(executor_outputs[1]);
      progress_bar.update("");
    }

    layer.set_states_all(std::move(states));

    return unbatch(output_batches, 1);
  }
}
