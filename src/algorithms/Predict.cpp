#include <libkann/algorithms/Predict.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/Graph.hpp>
#include <libkann/Executor.hpp>

namespace kann
{
  std::vector<tensor_t> predict(Layer& layer, Executor& executor, const std::vector<tensor_t>& inputs)
  {
    kann::Graph graph;

    // Forward pass
    size_t input_index = graph.add_vertex();
    kann::LayerDef::Info info;
    size_t output_index = layer.def->process(graph, info, input_index);

    // Executor
    Executor::Target target{
      .graph = std::move(graph),
      .input_indices  = {{input_index},  std::move(info.parameter_indices), std::move(info.input_state_indices)},
      .output_indices = {{output_index}, std::move(info.output_state_indices)}
    };

    // Compute
    std::vector<tensor_t> parameters = layer.get_parameters_all();
    std::vector<tensor_t> states     = layer.get_states_all();

    std::vector<tensor_t> outputs;
    outputs.reserve(inputs.size());
    for(const tensor_t& input : inputs)
    {
      auto executor_inputs = {{input}, parameters, std::move(states)};
      auto executor_outputs = executor.run(target, std::move(executor_inputs));
      outputs.push_back(std::move(executor_outputs[0].front()));
      states = std::move(executor_outputs[1]);
    }

    layer.set_states_all(std::move(states));

    return outputs;
  }
}
