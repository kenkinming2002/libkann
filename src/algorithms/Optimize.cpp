#include <libkann/algorithms/Optimize.hpp>

#include <libkann/algorithms/Helpers.hpp>

#include <libkann/Variable.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>

#include <libkann/Graph.hpp>
#include <libkann/Executor.hpp>
#include <libkann/Optimizer.hpp>

#include <libkann/Differentiate.hpp>
#include <libkann/operations/ScaleOperation.hpp>
#include <libkann/operations/SubtractOperation.hpp>

#include <fstream>
#include <map>

#include <range/v3/all.hpp>

namespace kann
{
  namespace
  {
    struct Option
    {
      layer_def_t def;
      Tag tag;
      optimizer_t optimizer;
      size_t batch_size;

      auto operator<=>(const Option& other) const = default;
    };

    // Graphs Creation
    static inline graph_t create_graph(const Option& option)
    {
      auto inputs           = make_input_variables(option.batch_size);
      auto expected_outputs = make_input_variables(option.batch_size);
      auto input_parameters = make_input_variables(option.def->parameters_all_count());
      auto input_states     = make_input_variables(option.def->states_all_count());

      auto predict_variables_input = PredictVariablesInput{
        .variables  = inputs,
        .parameters = input_parameters,
        .states     = input_states
      };
      auto predict_variables_output = make_predict_variables(*option.def, predict_variables_input);

      auto outputs       = predict_variables_output.variables;
      auto output_states = predict_variables_output.states;

      auto output_gradients = make_simple_output_gradients(outputs, expected_outputs);

      auto optimize_variables_input = OptimizeVariablesInput{
        .outputs = outputs,
        .output_gradients = output_gradients,
        .expected_outputs = expected_outputs,
        .parameters       = input_parameters
      };
      auto optimize_variables_output = make_optimize_variables(*option.def, *option.optimizer, option.tag, optimize_variables_input);

      auto output_parameters = optimize_variables_output.parameters;
      auto optimizer_input_states  = optimize_variables_output.input_states;
      auto optimizer_output_states = optimize_variables_output.output_states;

      return std::make_shared<const Graph>(
        std::vector{inputs, expected_outputs, input_parameters, input_states, optimizer_input_states},
        std::vector{outputs, output_parameters, output_states, optimizer_output_states}
      );
    }

    static inline graph_t get_graph(const BatchOptimizeInput& input)
    {
      assert(input.layer.def);
      assert(input.inputs.size() == input.expected_outputs.size());
      Option option = {
        .def        = input.layer.def,
        .tag        = input.tag,
        .optimizer  = input.optimizer,
        .batch_size = input.inputs.size()
      };

      static std::map<Option, graph_t> graphs;
      if(auto it = graphs.find(option); it != graphs.end())
        return it->second;

      if(auto [it, success] = graphs.emplace(option, create_graph(option)); success)
      {
        std::ofstream f;
        f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        f.open("output/optimize.dot");
        it->second->write_graphviz(f);
        return it->second;
      }

      assert(false && "Unreachable");
    }
  }

  OptimizeState create_optimize_state(const LayerDef& def, Tag tag, const Optimizer& optimizer)
  {
    OptimizeState optimize_state;

    auto parameter_tags  = def.parameter_tags();
    auto parameter_sizes = def.parameters_all_sizes();
    for(auto [parameter_tag, parameter_size] : ranges::views::zip(parameter_tags, parameter_sizes))
      if((parameter_tag & tag) == tag)
        optimize_state.values |= ranges::actions::push_back(optimizer.create_initial_states(parameter_size));
    return optimize_state;
  }

  BatchOptimizeOutput batch_optimize(const BatchOptimizeInput& input, OptimizeState& optimize_state, Executor& executor)
  {
    BatchOptimizeOutput output;

    auto graph = get_graph(input);

    {
      std::ofstream f;
      f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      f.open("output/optimize.dot");
      graph->write_graphviz(f);
    }

    auto inputs                 = input.inputs;
    auto expected_outputs       = input.expected_outputs;
    auto input_parameters       = input.layer.get_parameters_all();
    auto input_states           = input.layer.get_states_all();
    auto input_optimizer_states = optimize_state.values;

    auto executor_input  = std::vector{std::move(inputs), std::move(expected_outputs), std::move(input_parameters), std::move(input_states), std::move(input_optimizer_states)};
    auto executor_output = executor.process(graph, std::move(executor_input));

    auto outputs                 = std::move(executor_output[0]);
    auto output_parameters       = std::move(executor_output[1]);
    auto output_states           = std::move(executor_output[2]);
    auto output_optimizer_states = std::move(executor_output[3]);

    input.layer.set_states_all(std::move(output_states));
    input.layer.set_parameters_all(std::move(output_parameters));
    optimize_state.values = std::move(output_optimizer_states);
    output.outputs        = std::move(outputs);

    return output;
  }

  Task<void, OptimizeInfo> optimize(Layer& layer, Tag tag, optimizer_t optimizer,
      const std::vector<tensor_t>& inputs,
      const std::vector<tensor_t>& expected_outputs,
      size_t batch_size, Executor& executor)
  {
    OptimizeState state = create_optimize_state(*layer.def, tag, *optimizer);

    const auto& batches = ranges::views::zip(
        inputs           | ranges::views::chunk(batch_size) | ranges::views::transform([](auto&& r) { return std::forward<decltype(r)>(r) | ranges::to_vector; }),
        expected_outputs | ranges::views::chunk(batch_size) | ranges::views::transform([](auto&& r) { return std::forward<decltype(r)>(r) | ranges::to_vector; })
    );
    for(const auto& [inputs_batch, expected_output_batch] : batches)
    {
      kann::BatchOptimizeInput input = { .layer = layer, .tag= tag, .optimizer= optimizer, .inputs = inputs_batch, .expected_outputs = expected_output_batch };
      kann::BatchOptimizeOutput output = kann::batch_optimize(input, state, executor);
      for(const auto& [input, expected_output, output] : ranges::views::zip(input.inputs, input.expected_outputs, output.outputs))
      {
        OptimizeInfo info = {
          .input           = *input,
          .output          = *output,
          .expected_output = *expected_output,
          .cost            = (expected_output->asVector() - output->asVector()).squaredNorm()
        };
        co_yield info;
      }
    }
  }
}
