#include <libkann/Graph.hpp>

#include <stack>
#include <yaml-cpp/emitter.h>

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

  Graph::Graph(std::vector<std::shared_ptr<const Variable>> inputs, std::vector<std::shared_ptr<const Variable>> outputs)
  {
    assert(std::all_of(inputs.begin(),  inputs.end(),  [](const auto& v) { return (bool)v; }));
    assert(std::all_of(outputs.begin(), outputs.end(), [](const auto& v) { return (bool)v; }));

    // 1: Create nodes
    std::unordered_map<std::shared_ptr<const Variable>, size_t> indices_map;
    {
      // TODO: Use a stack instead of an unordered_set
      std::unordered_set<CRef<Variable>> open;
      open.insert(outputs.begin(), outputs.end());
      while(!open.empty())
      {
        auto variable = open.extract(open.begin()).value();
        if(indices_map.contains(variable))
          continue;

        size_t index = m_nodes.size();
        m_nodes.push_back(Node{.op = variable->op });
        indices_map.emplace(variable, index);
        open.insert(variable->inputs.begin(), variable->inputs.end());
      }
    }

    // 2: Initialize nodes
    for(const auto& [variable, index] : indices_map)
      m_nodes[index].input_indices = map(variable->inputs, [&,index=index](const auto& input) {
        size_t input_index = indices_map.at(input);
        m_nodes[input_index].output_indices.push_back(index);
        return input_index;
      });

    // 3: Save input/output index
    m_input_indices  = map(inputs, [&](const auto& input){ return indices_map.at(input); });
    m_output_indices = map(outputs, [&](const auto& output){ return indices_map.at(output); });
  }

  struct State
  {
    size_t index;
    size_t pos;
  };

  std::vector<size_t> Graph::topological_ordering() const
  {
    std::vector<size_t> ordering;
    std::vector<bool> visited;

    ordering.reserve(m_nodes.size());
    visited.resize(m_nodes.size(), false);
    {
      std::vector<State> state;
      state.reserve(m_nodes.size());

      for(size_t output_index : m_output_indices)
      {
        if(!visited[output_index])
        {
          visited[output_index] = true;
          state.push_back(State{.index = output_index, .pos = 0});
        }

        while(!state.empty())
        {
          auto& [index, pos] = state.back();
          const Node& node = m_nodes[index];
          if(node.input_indices.size() != pos)
          {
            size_t parent_index = node.input_indices[pos++];
            if(!visited[parent_index])
            {
              visited[parent_index] = true;
              state.push_back(State{.index = parent_index, .pos = 0});
            }
          }
          else
          {
            ordering.push_back(index);
            state.pop_back();
          }
        }
      }
    }

    return ordering;
  }
}
