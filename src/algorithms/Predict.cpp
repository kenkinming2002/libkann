#include <libkann/algorithms/Predict.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>

#include <libkann/Graph.hpp>
#include <libkann/Executor.hpp>

#include <map>

#include <range/v3/all.hpp>

namespace kann
{
  // TODO: Obsolete this via better executor API
  template<typename T, size_t N>
  static inline std::array<std::vector<T>, N> move_split(std::vector<T> data, std::array<size_t, N> sizes)
  {
    std::array<size_t, N> indices;
    std::exclusive_scan(sizes.begin(), sizes.end(), indices.begin(), 0);

    std::array<std::vector<T>, N> result;
    std::transform(indices.begin(), indices.end(), sizes.begin(), result.begin(), [&data](size_t index, size_t size) {
        assert(data.size() >= index + size);
        return std::vector(
            std::move_iterator(data.begin() + index),
            std::move_iterator(data.begin() + index + size)
            );
        });

    return result;
  }

  template<typename Arg, typename... Args>
  static inline auto move_split(Arg&& arg, Args&&... args) { return move_split(std::forward<Arg>(arg), std::array{std::forward<Args>(args)...}); }

  namespace
  {
    struct Option
    {
      std::shared_ptr<const LayerDef> def;
      size_t batch_size;

      auto operator<=>(const Option& other) const = default;
    };

    typedef std::vector<Tag> tags_t;
    typedef std::vector<std::shared_ptr<const Variable>> variables_t;

    static inline auto create_input_variables(size_t count)
    {
      return ranges::views::generate_n([]() { return std::make_shared<const Variable>(); }, count) | ranges::to_vector;
    }

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

      return std::make_shared<const Graph>(
          ranges::views::concat(inputs, input_parameters, input_states) | ranges::to_vector,
          ranges::views::concat(outputs, output_states)                 | ranges::to_vector
      );
    }

    static inline std::shared_ptr<const Graph> get_graph(const BatchPredictInput& input)
    {
      assert(input.layer.def);
      Option option = {
        .def = input.layer.def,
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

  BatchPredictOutput batch_predict(const BatchPredictInput& input, Executor& executor)
  {
    BatchPredictOutput output;

    auto graph = get_graph(input);

    auto inputs       = input.inputs;
    auto parameters   = input.layer.get_parameters_all();
    auto input_states = input.layer.get_states_all();

    auto executor_input  = ranges::views::concat(inputs, parameters, input_states) | ranges::to_vector;
    auto executor_output = executor.process(graph, executor_input);
    auto [outputs, output_states] =  move_split(executor_output,
        inputs.size(),
        input_states.size()
    );

    input.layer.set_states_all(std::move(output_states));
    output.outputs = std::move(outputs);

    return output;
  }

  Task<void, PredictInfo> predict(Layer& layer,
      const std::vector<std::shared_ptr<const Tensor>>& inputs,
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
