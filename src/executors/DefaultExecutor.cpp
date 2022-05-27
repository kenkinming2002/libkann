#include <libkann/executors/DefaultExecutor.hpp>

#include <libkann/Graph.hpp>
#include <libkann/Operation.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  std::vector<std::vector<tensor_t>> DefaultExecutor::process(graph_t graph, std::vector<std::vector<tensor_t>> inputs)
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
    for(const auto& [sub_input_indices, sub_inputs] : ranges::views::zip(graph->input_indices(), inputs))
      for(const auto& [input_index, input] : ranges::views::zip(sub_input_indices, sub_inputs))
        state.values[input_index] = input;

    // 4: Compute
    const auto& nodes = graph->nodes();
    for(size_t index : state.ordering)
    {
      const auto& node = nodes[index];
      if(state.values[index])
        continue;

      auto inputs = node.input_indices | ranges::views::transform([&](size_t input_index) {
          assert(state.values[input_index]);
          return state.values[input_index].get();
      }) | ranges::to_vector;
      assert(node.op);

      state.values[index] = node.op->process(std::move(inputs));
    }

    // 5: Outputs
    return graph->output_indices() | ranges::views::transform([&](const auto& sub_output_indices) {
        return sub_output_indices | ranges::views::transform([&](size_t output_index) {
            return state.values[output_index];
        }) | ranges::to_vector;
    }) | ranges::to_vector;
  }
}

