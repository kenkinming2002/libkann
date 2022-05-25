#include <libkann/Graph.hpp>

#include <range/v3/all.hpp>

#include <set>
#include <unordered_map>

namespace kann
{
  Graph::Graph(std::vector<std::vector<std::shared_ptr<const Variable>>> inputs,
               std::vector<std::vector<std::shared_ptr<const Variable>>> outputs)
  {
    assert(ranges::all_of(inputs  | ranges::views::join, [](const auto& v) { return (bool)v; }));
    assert(ranges::all_of(outputs | ranges::views::join, [](const auto& v) { return (bool)v; }));

    // 1: Create nodes
    std::unordered_map<std::shared_ptr<const Variable>, size_t> indices_map;
    {
      // TODO: Use a stack instead of an unordered_set
      std::set<std::shared_ptr<const Variable>> open;
      ranges::actions::insert(open, inputs | ranges::views::join);
      ranges::actions::insert(open, outputs | ranges::views::join);

      while(!open.empty())
      {
        auto variable = open.extract(open.begin()).value();
        if(indices_map.contains(variable))
          continue;

        size_t index = m_nodes.size();
        m_nodes.push_back(Node{.op = variable->op });
        indices_map.emplace(variable, index);
        ranges::actions::insert(open, variable->inputs);
      }
    }

    // 2: Create edges
    for(const auto& [output, output_index] : indices_map)
    {
      m_nodes[output_index].input_indices = output->inputs
        | ranges::views::transform([&](const auto& input) { return indices_map.at(input); })
        | ranges::to_vector;

      for(size_t input_index : m_nodes[output_index].input_indices)
        m_nodes[input_index].output_indices.push_back(output_index);
    }

    // 3: Save input/output indices
    m_input_indices = inputs | ranges::views::transform([&](const auto& sub_inputs) {
        return sub_inputs | ranges::views::transform([&](const auto& input){
            return indices_map.at(input);
        }) | ranges::to_vector;
    }) | ranges::to_vector;

    m_output_indices = outputs | ranges::views::transform([&](const auto& sub_outputs) {
        return sub_outputs | ranges::views::transform([&](const auto& output){
            return indices_map.at(output);
        }) | ranges::to_vector;
    }) | ranges::to_vector;
  }

  static inline void topological_ordering_impl(const std::vector<Graph::Node>& nodes, std::vector<size_t>& ordering, std::vector<bool>& visited, size_t index)
  {
    if(visited[index])
      return;

    visited[index] = true;
    for(size_t parent_index : nodes[index].input_indices)
      topological_ordering_impl(nodes, ordering, visited, parent_index);

    ordering.push_back(index);
  }

  std::vector<size_t> Graph::topological_ordering() const
  {
    std::vector<size_t> ordering;
    std::vector<bool> visited;

    ordering.reserve(m_nodes.size());
    visited.resize(m_nodes.size(), false);

    for(size_t index = 0; index<m_nodes.size(); ++index)
      topological_ordering_impl(m_nodes, ordering, visited, index);

    return ordering;
  }
}
