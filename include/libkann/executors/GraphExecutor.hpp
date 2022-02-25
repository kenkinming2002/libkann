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
    void addInput(std::string name, std::vector<CRef<Variable>> variables) override;
    void addOutput(std::string name, std::vector<CRef<Variable>> variables) override;

  public:
    void build() override;

  public:
    void write_graphviz(std::ostream& os) const override;


  protected:
    const auto& inputVertices(std::string name)  { return m_inputVerticesMap.at(name); }
    const auto& outputVertices(std::string name) { return m_outputVerticesMap.at(name); }
    const auto& graph() { return m_graph; }

  private:
    std::unordered_map<std::string, std::vector<CRef<Variable>>> m_inputVariablesMap;
    std::unordered_map<std::string, std::vector<CRef<Variable>>> m_outputVariablesMap;

  private:
    graph_type m_graph;
    std::unordered_map<std::string, std::vector<vertex_type>> m_inputVerticesMap;
    std::unordered_map<std::string, std::vector<vertex_type>> m_outputVerticesMap;
  };
}
