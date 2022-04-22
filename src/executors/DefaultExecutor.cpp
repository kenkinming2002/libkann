#include <libkann/executors/DefaultExecutor.hpp>

#include <libkann/Operation.hpp>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/topological_sort.hpp>

#include <iterator>

namespace kann
{
  template<typename From, typename F>
  static std::vector<std::result_of_t<F(const From&)>> map(const std::vector<From>& input, F f)
  {
    std::vector<std::result_of_t<F(const From&)>> output;
    output.reserve(input.size());
    for(const auto& data : input)
      output.push_back(f(data));

    return output;
  }

  std::vector<std::shared_ptr<const Tensor>> DefaultExecutor::process(std::shared_ptr<const Graph> graph, std::vector<std::shared_ptr<const Tensor>> inputs)
  {
    // 1: Find or Create state
    auto it = m_states.find(graph);
    if(it == m_states.end())
    {
      State state;
      state.ordering = graph->topological_ordering();
      state.values.resize(state.ordering.size());

      it = m_states.emplace(graph, std::move(state)).first;
    }

    State& state = it->second;

    // 2: Reset
    std::fill(state.values.begin(), state.values.end(), nullptr);

    // 3: Inputs
    {
      const auto& input_indices  = graph->input_indices();
      assert(input_indices.size() == inputs.size());
      for(size_t i=0; i<input_indices.size(); ++i)
        state.values[input_indices[i]] = inputs[i];
    }

    // 4: Compute
    const auto& nodes = graph->nodes();
    for(size_t index : state.ordering)
    {
      const auto& node = nodes[index];
      if(state.values[index])
        continue;

      auto inputs = map(node.input_indices, [&](size_t input_index) -> const Tensor* {
          assert(state.values[input_index]);
          return state.values[input_index].get();
      });
      assert(node.op);
      state.values[index] = node.op->process(std::move(inputs));
    }

    // 5: Outputs
    return map(graph->output_indices(), [&](size_t output_index) {
        return state.values[output_index];
    });
  }
}

