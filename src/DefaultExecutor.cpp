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

  void DefaultExecutor::addInput(std::string name, std::vector<std::shared_ptr<const Variable>> variables)
  {
    auto [_, success] = m_inputVariablesMap.emplace(std::move(name), std::move(variables));
    assert(success);
  }

  void DefaultExecutor::addOutput(std::string name, std::vector<std::shared_ptr<const Variable>> variables)
  {
    auto [_, success] = m_outputVariablesMap.emplace(std::move(name), std::move(variables));
    assert(success);
  }

  template<typename Callback>
  static void walk(const std::shared_ptr<const Variable>& variable, const Callback& callback)
  {
    if(!callback(variable))
      return;

    for(const auto& input : variable->inputs)
      walk(input, callback);
  }

  void DefaultExecutor::build()
  {
    // 1: Create vertices
    std::unordered_map<std::shared_ptr<const Variable>, vertex_type> verticesMap;
    for(const auto& [name, outputVariables] : m_outputVariablesMap)
      for(const auto& outputVariable : outputVariables)
        walk(outputVariable, [&](const auto& variable){
          auto it = verticesMap.find(variable);
          if(it != verticesMap.end())
            return false;

          auto vertex = boost::add_vertex(Node{.variable = variable}, m_graph);
          verticesMap.emplace(variable, vertex);
          return true;
        });

    // 2: Create edges
    for(const auto& [variable, vertex] : verticesMap)
      for(size_t i=0; i<variable->inputs.size(); ++i)
      {
        const auto& inputVariable = variable->inputs[i];
        const auto& inputVertex = verticesMap.at(inputVariable);
        boost::add_edge(inputVertex, vertex, Connection{.id = i}, m_graph);
      }

    // 3: Save necessary association
    for(const auto& [name, inputVariables] : m_inputVariablesMap)
    {
      std::vector<vertex_type> inputVertices;
      inputVertices.reserve(inputVariables.size());
      for(const auto& inputVariable : inputVariables)
        inputVertices.push_back(verticesMap.at(inputVariable));

      m_inputVerticesMap.emplace(name, std::move(inputVertices));
    }

    for(const auto& [name, outputVariables] : m_outputVariablesMap)
    {
      std::vector<vertex_type> outputVertices;
      outputVertices.reserve(outputVariables.size());
      for(const auto& outputVariable : outputVariables)
        outputVertices.push_back(verticesMap.at(outputVariable));

      m_outputVerticesMap.emplace(name, std::move(outputVertices));
    }

    boost::topological_sort(m_graph, std::back_inserter(m_ordering));
    std::reverse(m_ordering.begin(), m_ordering.end());
  }

  void DefaultExecutor::input(std::string name, std::vector<std::shared_ptr<const Tensor>> input)
  {
    if(!m_dirty)
    {
      // Clear cache
      for(auto [it, end] = boost::vertices(m_graph); it != end; ++it)
      {
        Node& node = m_graph[*it];
        node.data.reset();
      }
      m_dirty = true;
    }

    const auto& inputVertices = m_inputVerticesMap.at(name);

    assert(inputVertices.size() == input.size());
    for(size_t i=0; i<inputVertices.size(); ++i)
    {
      Node& inputNode = m_graph[inputVertices[i]];
      inputNode.data = input[i];
    }
  }

  std::vector<std::shared_ptr<const Tensor>> DefaultExecutor::output(std::string name)
  {
    if(m_dirty)
    {
      // Evaluate
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
      m_dirty = false;
    }

    const auto& outputVertices = m_outputVerticesMap.at(name);

    std::vector<std::shared_ptr<const Tensor>> outputs(outputVertices.size());
    for(size_t i=0; i<outputVertices.size(); ++i)
    {
      Node& outputNode = m_graph[outputVertices[i]];
      outputs[i] = outputNode.data;
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

  std::unique_ptr<Executor> makeDefaultExecutor()
  {
    return std::make_unique<DefaultExecutor>();
  }
}

