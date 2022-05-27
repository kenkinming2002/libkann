#pragma once

#include <libkann/Types.hpp>

#include <vector>

namespace kann
{
  /* A computational graph */
  class Graph
  {
  public:
    Graph(std::vector<std::vector<variable_t>> inputs,
          std::vector<std::vector<variable_t>> outputs);

  public:
    struct Node
    {
      // TODO: Rename to parent and child indices
      std::vector<size_t> input_indices;
      std::vector<size_t> output_indices; // Do I need it

      operation_t op;
    };

  public:
    size_t size() const { return m_nodes.size(); }

  public:
    const auto& input_indices() const { return m_input_indices; }
    const auto& output_indices() const { return m_output_indices; }

  public:
    const auto& nodes() const { return m_nodes; }

  public:
    std::vector<size_t> topological_ordering() const;

  private:
    std::vector<Node> m_nodes;
    std::vector<std::vector<size_t>> m_input_indices;
    std::vector<std::vector<size_t>> m_output_indices;
  };
}
