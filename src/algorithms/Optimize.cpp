#include <libkann/algorithms/Optimize.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>

#include <libkann/Graph.hpp>
#include <libkann/Executor.hpp>
#include <libkann/Optimizer.hpp>

#include <libkann/Differentiate.hpp>
#include <libkann/operations/MultiplyOperation.hpp>
#include <libkann/operations/SubtractOperation.hpp>

#include <map>

#include <range/v3/all.hpp>

namespace kann
{
  namespace
  {
    struct Option
    {
      std::shared_ptr<const LayerDef> def;
      Tag tag;
      std::shared_ptr<const Optimizer> optimizer;
      size_t batch_size;

      auto operator<=>(const Option& other) const = default;
    };

    typedef std::vector<Tag> tags_t;
    typedef std::vector<std::shared_ptr<const Variable>> variables_t;

    static inline auto create_input_variables(size_t count)
    {
      return ranges::views::generate_n([]() { return std::make_shared<const Variable>(); }, count) | ranges::to_vector;
    }

    // Graphs Creation
    static inline std::shared_ptr<const Graph> create_graph(const Option& option)
    {
      // 1: All kinds of inputs
      variables_t input_parameters = create_input_variables(option.def->parameters_all_count());
      variables_t input_states     = create_input_variables(option.def->states_all_count());

      // 2: Prediction step -> create new states
      variables_t inputs, outputs;
      variables_t output_states = input_states;

      inputs = create_input_variables(option.batch_size);
      outputs.reserve(option.batch_size);
      for(const auto& input : inputs)
      {
        auto process_input = LayerDef::ProcessInput{
          .variable   = input,
          .parameters = input_parameters,
          .states     = output_states
        };

        auto process_output = option.def->process(process_input);
        outputs.push_back(process_output.variable);
        output_states = process_output.states;
      }

      // 3: Differentiation step
      variables_t expected_outputs = create_input_variables(option.batch_size);
      variables_t output_gradients = ranges::views::transform(outputs, expected_outputs, [](const auto& output, const auto& expected_output) {
        auto diff = Variable::apply(SubtractOperation(), {output, expected_output});
        return Variable::apply(MultiplyOperation(2.0), {std::move(diff)});
      }) | ranges::to_vector;
      auto gradients_map = differentiate(outputs, output_gradients);

      // 4: Optimization step
      tags_t tags = option.def->parameters_tags();
      variables_t optimizer_input_states, optimizer_output_states;
      variables_t output_parameters = input_parameters;
      for(const auto& [i, input_parameter, tag] : ranges::views::zip(ranges::views::ints, input_parameters, tags))
      {
        if((tag & option.tag) == option.tag)
        {
          auto process_input = Optimizer::ProcessInput{
            .parameter = input_parameter,
            .gradient  = gradients_map.at(input_parameter)
          };

          auto process_output = option.optimizer->process(process_input);
          output_parameters[i] = process_output.parameter;
          optimizer_input_states  |= ranges::actions::push_back(process_output.input_states);
          optimizer_output_states |= ranges::actions::push_back(process_output.output_states);
        }
      }

      return std::make_shared<const Graph>(
        std::vector{inputs, expected_outputs, input_parameters, input_states, optimizer_input_states},
        std::vector{outputs, output_parameters, output_states, optimizer_output_states}
      );
    }

    static inline std::shared_ptr<const Graph> get_graph(const BatchOptimizeInput& input)
    {
      assert(input.layer.def);
      assert(input.inputs.size() == input.expected_outputs.size());
      Option option = {
        .def        = input.layer.def,
        .tag        = input.tag,
        .optimizer  = input.optimizer,
        .batch_size = input.inputs.size()
      };

      static std::map<Option, std::shared_ptr<const Graph>> graphs;
      if(auto it = graphs.find(option); it != graphs.end())
        return it->second;

      if(auto [it, success] = graphs.emplace(option, create_graph(option)); success)
        return it->second;

      assert(false && "Unreachable");
    }
  }

  OptimizeState create_optimize_state(const LayerDef& def, Tag tag, const Optimizer& optimizer)
  {
    OptimizeState optimize_state;

    auto parameter_tags  = def.parameters_tags();
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

  Task<void, OptimizeInfo> optimize(Layer& layer, Tag tag, std::shared_ptr<const Optimizer> optimizer,
      const std::vector<std::shared_ptr<const Tensor>>& inputs,
      const std::vector<std::shared_ptr<const Tensor>>& expected_outputs,
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
