#pragma once

#include <libkann/Executor.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <optional>

namespace kann
{
  class DefaultExecutor : public Executor
  {
  public:
    void addInput(std::string name, std::vector<std::shared_ptr<const Variable>> variables) override;
    void addOutput(std::string name, std::vector<std::shared_ptr<const Variable>> variables) override;

  public:
    void build() override;

  public:
    void input(std::string name, std::vector<std::shared_ptr<const Tensor>> input) override;
    std::vector<std::shared_ptr<const Tensor>> output(std::string name) override;

  public:
    void write_graphviz(std::ostream& os) const override;

  private:
    std::unordered_map<std::string, std::vector<std::shared_ptr<const Variable>>> m_inputVariablesMap;
    std::unordered_map<std::string, std::vector<std::shared_ptr<const Variable>>> m_outputVariablesMap;

  private:
    struct Node
    {
      std::shared_ptr<const Variable> variable;
      std::shared_ptr<const Tensor> data;

      std::optional<std::string> name;
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
    std::unordered_map<std::string, std::vector<vertex_type>> m_inputVerticesMap;
    std::unordered_map<std::string, std::vector<vertex_type>> m_outputVerticesMap;
    std::vector<vertex_type> m_ordering;

  private:
    bool m_dirty;
  };
}
