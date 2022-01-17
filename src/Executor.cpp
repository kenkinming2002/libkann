#include <libkann/Executor.hpp>

#include <boost/graph/graphviz.hpp>

namespace kann
{
  template<typename Callback>
  static void walk(const std::shared_ptr<const Variable>& variable, const Callback& callback)
  {
    if(!callback(variable))
      return;

    for(const auto& input : variable->inputs)
      walk(input, callback);
  }

  Executor::Executor(std::vector<std::shared_ptr<const Variable>> inputs, std::vector<std::shared_ptr<const Variable>> outputs)
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
  }

  std::vector<Tensor> Executor::evaluate(std::vector<Tensor> inputs)
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

    // Evaluate
    for(size_t i=0; i<m_outputVertices.size(); ++i)
      evaluate(m_outputVertices[i]);

    // Output
    std::vector<Tensor> outputs(m_outputVertices.size());
    for(size_t i=0; i<m_outputVertices.size(); ++i)
    {
      Node& outputNode = m_graph[m_outputVertices[i]];
      outputs[i] = std::move(*outputNode.data);
    }
    return outputs;
  }

  void Executor::evaluate(vertex_type vertex)
  {
    Node& node = m_graph[vertex];
    if(node.data)
      return;

    std::vector<const Tensor*> inputs_ptr(node.variable->inputs.size());
    for(auto [it, end] = boost::in_edges(vertex, m_graph); it != end; ++it)
    {
      const edge_type edge = *it;
      const vertex_type inputVertex = boost::source(edge, m_graph);

      const Node& inputNode = m_graph[inputVertex];
      const Connection& connection = m_graph[edge];

      evaluate(inputVertex);
      assert(inputNode.data);
      inputs_ptr[connection.id] = &(*inputNode.data);
    }

    std::vector<std::reference_wrapper<const Tensor>> inputs;
    inputs.reserve(inputs_ptr.size());
    std::transform(
      inputs_ptr.begin(), inputs_ptr.end(),
      std::back_inserter(inputs),
      [](const Tensor* t) -> std::reference_wrapper<const Tensor> { return *t; }
    );

    Tensor output = node.variable->op->process(inputs);
    node.data = std::move(output);
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

  void Executor::write_graphviz(std::ostream& os) const
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

