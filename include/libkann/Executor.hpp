#pragma once

#include <libkann/Variable.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

namespace kann
{
  class Executor
  {
  public:
    Executor() = default;
    Executor(std::vector<std::shared_ptr<const Variable>> inputs, std::vector<std::shared_ptr<const Variable>> outputs);

  public:
    std::vector<std::shared_ptr<const Tensor>> evaluate(std::vector<std::shared_ptr<const Tensor>> inputs);

  public:
    void write_graphviz(std::ostream& os) const;

  private:
    struct Node
    {
      std::shared_ptr<const Variable> variable;
      std::shared_ptr<const Tensor> data;
    };

    struct Connection
    {
      size_t id;
    };

  private:
    typedef boost::adjacency_list<
      boost::vecS, boost::vecS,
      boost::bidirectionalS,
      Node, Connection
    > graph_type;
    typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_type;
    typedef boost::graph_traits<graph_type>::edge_descriptor edge_type;

  private:
    graph_type m_graph;
    std::vector<vertex_type> m_ordering;
    std::vector<vertex_type> m_inputVertices;
    std::vector<vertex_type> m_outputVertices;
  };
}
