#pragma once

#include <libkann/Variable.hpp>
#include <libkann/Operation.hpp>
#include <libkann/Tensor.hpp>

#include <vector>
#include <memory>

namespace kann
{
  /* A computational graph */
  class Graph
  {
  public:
    Graph(std::vector<std::vector<std::shared_ptr<const Variable>>> inputs,
          std::vector<std::vector<std::shared_ptr<const Variable>>> outputs);

  public:
    struct Node
    {
      // TODO: Rename to parent and child indices
      std::vector<size_t> input_indices;
      std::vector<size_t> output_indices; // Do I need it

      std::shared_ptr<const Operation> op;
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
