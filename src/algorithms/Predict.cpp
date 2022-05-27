#include <libkann/algorithms/Predict.hpp>

#include <libkann/Variable.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>

#include <libkann/Graph.hpp>
#include <libkann/Executor.hpp>

#include <map>

#include <range/v3/all.hpp>

namespace kann
{
  namespace
  {
    struct Option
    {
      layer_def_t def;
      size_t batch_size;

      auto operator<=>(const Option& other) const = default;
    };

    typedef std::vector<Tag> tags_t;
    typedef std::vector<variable_t> variables_t;

    static inline auto create_input_variables(size_t count)
    {
      return ranges::views::generate_n([]() { return std::make_shared<const Variable>(); }, count) | ranges::to_vector;
    }

    static inline graph_t create_graph(const Option& option)
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

      return std::make_shared<const Graph>(
          std::vector{inputs, input_parameters, input_states},
          std::vector{outputs, output_states}
      );
    }

    static inline graph_t get_graph(const BatchPredictInput& input)
    {
      assert(input.layer.def);
      Option option = {
        .def = input.layer.def,
        .batch_size = input.inputs.size()
      };

      static std::map<Option, graph_t> graphs;
      if(auto it = graphs.find(option); it != graphs.end())
        return it->second;

      if(auto [it, success] = graphs.emplace(option, create_graph(option)); success)
        return it->second;

      assert(false && "Unreachable");
    }
  }

  BatchPredictOutput batch_predict(const BatchPredictInput& input, Executor& executor)
  {
    BatchPredictOutput output;

    auto graph = get_graph(input);

    auto inputs       = input.inputs;
    auto parameters   = input.layer.get_parameters_all();
    auto input_states = input.layer.get_states_all();

    auto executor_input  = std::vector{std::move(inputs), std::move(parameters), std::move(input_states)};
    auto executor_output = executor.process(graph, std::move(executor_input));

    auto outputs       = std::move(executor_output[0]);
    auto output_states = std::move(executor_output[1]);

    input.layer.set_states_all(std::move(output_states));
    output.outputs = std::move(outputs);

    return output;
  }

  Task<void, PredictInfo> predict(Layer& layer,
      const std::vector<tensor_t>& inputs,
      size_t batch_size, Executor& executor)
  {
    const auto& batches = inputs | ranges::views::chunk(batch_size) | ranges::views::transform([](auto&& r) { return std::forward<decltype(r)>(r) | ranges::to_vector; });
    for(const auto& inputs_batch : batches)
    {
      kann::BatchPredictInput batch_predict_input = { .layer = layer, .inputs = inputs_batch };
      kann::BatchPredictOutput batch_predict_output = kann::batch_predict(batch_predict_input, executor);
      for(const auto& [input, output] : ranges::views::zip(inputs_batch, batch_predict_output.outputs))
      {
        PredictInfo info = {
          .input           = *input,
          .output          = *output
        };
        co_yield info;
      }
    }
  }
}
