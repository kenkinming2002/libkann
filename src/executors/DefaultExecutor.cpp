#include <libkann/executors/DefaultExecutor.hpp>

#include <libkann/Graph.hpp>
#include <libkann/Operation.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  static inline void set_value(const Graph& graph, std::vector<std::optional<Tensor>>& values, size_t index, Tensor value)
  {
    assert(!graph.vertex(index).in_edge_index);
    assert(!values[index]);
    values[index] = value;
  }

  static inline Tensor get_value(const Graph& graph, std::vector<std::optional<Tensor>>& values, size_t index)
  {
    if(!values[index])
    {
      // Compute recursively
      const Graph::Vertex& vertex = graph.vertex(index);

      assert(vertex.in_edge_index);
      const Graph::Edge& edge = graph.edge(*vertex.in_edge_index);
      std::vector<Tensor> inputs = edge.input_indices
        | ranges::views::transform([&graph, &values](size_t input_index) { return get_value(graph, values, input_index); })
        | ranges::to_vector;

      std::vector<Tensor> outputs = edge.op->process(std::move(inputs));
      for(const auto& [output_index, output] : ranges::views::zip(edge.output_indices, outputs))
      {
        assert(!values[output_index]);
        values[output_index] = output;
      }
    }

    assert(values[index]);
    return *values[index];
  }

  std::vector<std::vector<Tensor>> DefaultExecutor::run(const Target& target, std::vector<std::vector<Tensor>> inputs) const
  {
    std::vector<std::optional<Tensor>> values;
    values.resize(target.graph.vertices_count());

    for(const auto& [i, sub_input_indices] : ranges::views::enumerate(target.input_indices))
      for(const auto& [j, input_index] : ranges::views::enumerate(sub_input_indices))
        set_value(target.graph, values, input_index, inputs[i][j]);

    return target.output_indices | ranges::views::transform([&](const std::vector<size_t>& sub_output_indices) {
      return sub_output_indices | ranges::views::transform([&](size_t output_index) {
        return get_value(target.graph, values, output_index);
      }) | ranges::to_vector;
    }) | ranges::to_vector;
  }
}

