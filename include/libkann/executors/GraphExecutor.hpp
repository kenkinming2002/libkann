#pragma once

#include <libkann/Executor.hpp>

#include <boost/graph/adjacency_list.hpp>

namespace kann
{
  /* Base class for graph-based executor */
  class GraphExecutor : public Executor
  {
  protected:
    struct Node
    {
      size_t inputCount; // TODO: Retrive that from Operation
      CRef<Operation> op;
      CRef<Tensor> value;
    };

    struct Connection
    {
      size_t i;
    };

  protected:
    typedef boost::adjacency_list<
      boost::vecS, boost::vecS,
      boost::bidirectionalS,
      Node, Connection
    > graph_type;
    typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_type;
    typedef boost::graph_traits<graph_type>::edge_descriptor edge_type;

  public:
    void build(std::vector<CRef<Variable>> inputs, std::vector<CRef<Variable>> outputs) override final;
    std::vector<CRef<Tensor>> process(std::vector<CRef<Tensor>> inputs) override;

  public:
    void write_graphviz(std::ostream& os) const override;

  protected:
    const auto& graph() const { return m_graph; }
    auto& graph() { return m_graph; }

    const auto& inputVertices() const { return m_inputVertices; }
    const auto& outputVertices() const { return m_outputVertices; }

  protected:
    virtual void build() = 0;
    virtual void compute() = 0;

  private:
    graph_type m_graph;
    std::vector<vertex_type> m_inputVertices;
    std::vector<vertex_type> m_outputVertices;
  };
}
