#include <libkann/executors/GraphExecutor.hpp>

#include <libkann/Operation.hpp>

#include <boost/graph/graphviz.hpp>

namespace kann
{
  void GraphExecutor::addInput(std::string name, std::vector<std::shared_ptr<const Variable>> variables)
  {
    auto [_, success] = m_inputVariablesMap.emplace(std::move(name), std::move(variables));
    assert(success);
  }

  void GraphExecutor::addOutput(std::string name, std::vector<std::shared_ptr<const Variable>> variables)
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

  void GraphExecutor::build()
  {
    // 1: Create vertices
    std::unordered_map<std::shared_ptr<const Variable>, vertex_type> verticesMap;
    for(const auto& [name, outputVariables] : m_outputVariablesMap)
      for(const auto& outputVariable : outputVariables)
        walk(outputVariable, [&](const auto& variable){
          auto it = verticesMap.find(variable);
          if(it != verticesMap.end())
            return false;

          auto vertex = boost::add_vertex(Node{
            .inputCount = variable->inputs.size(),
            .op = variable->op
          }, m_graph);

          verticesMap.emplace(variable, vertex);
          return true;
        });

    // 2: Create edges
    for(const auto& [variable, vertex] : verticesMap)
      for(size_t i=0; i<variable->inputs.size(); ++i)
      {
        const auto& inputVariable = variable->inputs[i];
        const auto& inputVertex = verticesMap.at(inputVariable);
        boost::add_edge(inputVertex, vertex, Connection{.i = i}, m_graph);
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

  void GraphExecutor::write_graphviz(std::ostream& os) const
  {
    auto vertexWriter = [this](std::ostream& os, vertex_type vertex){
      const Node& node = m_graph[vertex];
      os << "[label=\"";

      if(node.op)
      {
        const Operation& op = *node.op;
        os << "op=" << demangle(typeid(op).name()).substr(0, 20) << "\\n";
      }

      os << "\"]";
    };

    auto edgeWriter = [this](std::ostream& os, edge_type edge){
      const Connection& connection = m_graph[edge];
      os << "[label=\"";
      os << "i=" << connection.i << "\\n";
      os << "\"]";
    };

    boost::write_graphviz(os, m_graph, vertexWriter, edgeWriter);
  }
}
