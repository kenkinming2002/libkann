#include <libkann/DefaultExecutor.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/topological_sort.hpp>

#include <iterator>

namespace kann
{
  class DefaultExecutor : public Executor
  {
  public:
    DefaultExecutor() = default;
    DefaultExecutor(std::vector<std::shared_ptr<const Variable>> inputs, std::vector<std::shared_ptr<const Variable>> outputs);

  public:
    std::vector<std::shared_ptr<const Tensor>> evaluate(std::vector<std::shared_ptr<const Tensor>> inputs) override;

  public:
    void write_graphviz(std::ostream& os) const override;

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

  std::unique_ptr<Executor> makeDefaultExecutor(std::vector<std::shared_ptr<const Variable>> inputs, std::vector<std::shared_ptr<const Variable>> outputs)
  {
    return std::make_unique<DefaultExecutor>(std::move(inputs), std::move(outputs));
  }


  template<typename Callback>
  static void walk(const std::shared_ptr<const Variable>& variable, const Callback& callback)
  {
    if(!callback(variable))
      return;

    for(const auto& input : variable->inputs)
      walk(input, callback);
  }

  DefaultExecutor::DefaultExecutor(std::vector<std::shared_ptr<const Variable>> inputs, std::vector<std::shared_ptr<const Variable>> outputs)
  {
    std::unordered_map<std::shared_ptr<const Variable>, vertex_type> verticesMap;
    for(const auto& output : outputs)
      walk(output, [&](const auto& variable){
        auto it = verticesMap.find(variable);
        if(it != verticesMap.end())
          return false;

        auto vertex = boost::add_vertex(Node{.variable = variable}, m_graph);
        verticesMap.emplace(variable, vertex);
        return true;
      });

    for(const auto& [variable, vertex] : verticesMap)
      for(size_t i=0; i<variable->inputs.size(); ++i)
      {
        const auto& inputVariable = variable->inputs[i];
        const auto& inputVertex = verticesMap.at(inputVariable);
        boost::add_edge(inputVertex, vertex, Connection{.id = i}, m_graph);
      }

    m_inputVertices.reserve(inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(m_inputVertices), [&](const auto& input){
        return verticesMap.at(input);
    });

    m_outputVertices.reserve(outputs.size());
    std::transform(outputs.begin(), outputs.end(), std::back_inserter(m_outputVertices), [&](const auto& output){
        return verticesMap.at(output);
    });

    boost::topological_sort(m_graph, std::back_inserter(m_ordering));
    std::reverse(m_ordering.begin(), m_ordering.end());
  }

  std::vector<std::shared_ptr<const Tensor>> DefaultExecutor::evaluate(std::vector<std::shared_ptr<const Tensor>> inputs)
  {
    // Clear everything
    for(auto [it, end] = boost::vertices(m_graph); it != end; ++it)
    {
      Node& node = m_graph[*it];
      node.data.reset();
    }

    // Input
    for(size_t i=0; i<m_inputVertices.size(); ++i)
    {
      Node& inputNode = m_graph[m_inputVertices[i]];
      inputNode.data = std::move(inputs[i]);
    }

    for(vertex_type vertex : m_ordering)
    {
      Node& node = m_graph[vertex];
      if(node.data)
        continue; // One of the input vertex

      std::vector<std::shared_ptr<const Tensor>> inputs(node.variable->inputs.size());
      for(auto [it, end] = boost::in_edges(vertex, m_graph); it != end; ++it)
      {
        const edge_type edge = *it;
        const vertex_type inputVertex = boost::source(edge, m_graph);

        const Node& inputNode = m_graph[inputVertex];
        const Connection& connection = m_graph[edge];

        assert(inputNode.data);
        inputs[connection.id] = inputNode.data;
      }

      node.data = node.variable->op->process(inputs);
    }

    // Output
    std::vector<std::shared_ptr<const Tensor>> outputs(m_outputVertices.size());
    for(size_t i=0; i<m_outputVertices.size(); ++i)
    {
      Node& outputNode = m_graph[m_outputVertices[i]];
      outputs[i] = std::move(outputNode.data);
    }
    return outputs;
  }

  static std::string demangle(const char* mangledName)
  {
    int status;
    char* demangledName = abi::__cxa_demangle(mangledName, nullptr, 0, &status);
    if(status != 0)
      return mangledName;

    std::string result = demangledName;
    free(demangledName);
    return result;
  }

  void DefaultExecutor::write_graphviz(std::ostream& os) const
  {
    auto vertexWriter = [this](std::ostream& os, vertex_type vertex){
      const Node& node = m_graph[vertex];
      const Variable& variable = *node.variable;
      if(!variable.op)
        return;

      const Operation& op = *variable.op;

      os << "[label=\"";
      os << "op=" << demangle(typeid(op).name()).substr(0, 20) << "\\n";
      os << "\"]";
    };

    auto edgeWriter = [this](std::ostream& os, edge_type edge){
      const Connection& connection = m_graph[edge];
      os << "[label=\"";
      os << "id=" << connection.id << "\\n";
      os << "\"]";
    };
    boost::write_graphviz(os, m_graph, vertexWriter, edgeWriter);
  }
}

