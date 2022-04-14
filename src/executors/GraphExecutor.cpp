#include <libkann/executors/GraphExecutor.hpp>

#include <libkann/Operation.hpp>

#include <boost/graph/graphviz.hpp>

namespace kann
{
  template<typename From, typename F>
  static std::vector<std::result_of_t<F(const From&)>> map(const std::vector<From>& input, F f)
  {
    std::vector<std::result_of_t<F(const From&)>> output;
    output.reserve(input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(output), f);
    return output;
  }

  void GraphExecutor::build(std::vector<CRef<Variable>> inputs, std::vector<CRef<Variable>> outputs)
  {
    assert(std::all_of(inputs.begin(),  inputs.end(),  [](const auto& v) { return (bool)v; }));
    assert(std::all_of(outputs.begin(), outputs.end(), [](const auto& v) { return (bool)v; }));

    // 1: Create vertices
    std::unordered_map<CRef<Variable>, vertex_type> verticesMap;
    {
      std::unordered_set<CRef<Variable>> open;
      //open.insert(inputs.begin(),  inputs.end()); // A Hack
      open.insert(outputs.begin(), outputs.end());
      while(!open.empty())
      {
        auto variable = open.extract(open.begin()).value();
        if(verticesMap.contains(variable))
          continue;

        auto node = Node{.inputCount = variable->inputs.size(), .op = variable->op};
        auto vertex = boost::add_vertex(std::move(node), m_graph);
        auto [_, success] = verticesMap.emplace(variable, vertex);
        assert(success);

        open.insert(variable->inputs.begin(), variable->inputs.end());
      }
    }

    // 2: Create edges
    for(const auto& [variable, vertex] : verticesMap)
      for(size_t i=0; i<variable->inputs.size(); ++i)
      {
        const auto& inputVariable = variable->inputs[i];
        const auto& inputVertex = verticesMap.at(inputVariable);
        boost::add_edge(inputVertex, vertex, Connection{.i = i}, m_graph);
      }

    // 3: Save necessary association
    m_inputVertices = map(inputs, [&verticesMap](const auto& input){
      return verticesMap.at(input);
    });

    m_outputVertices = map(outputs, [&verticesMap](const auto& output){
      return verticesMap.at(output);
    });

    // 4: Allow subclass to build
    this->build();
  }

  std::vector<CRef<Tensor>> GraphExecutor::process(std::vector<CRef<Tensor>> inputs)
  {
    assert(inputs.size() == m_inputVertices.size());
    assert(std::all_of(inputs.begin(), inputs.end(), [](const auto& input) { return (bool)input; }));

    // Clear
    for(auto [it, end] = boost::vertices(m_graph); it != end; ++it)
      m_graph[*it].value.reset();

    // Inputs
    {
      size_t i = 0;
      for(const auto vertex : m_inputVertices)
      {
        m_graph[vertex].value = inputs[i++];
      }
    }

    // Compute
    this->compute();

    // Outputs
    {
      std::vector<CRef<Tensor>> outputs(m_outputVertices.size());

      size_t i = 0;
      for(const auto vertex : m_outputVertices)
        outputs[i++] = m_graph[vertex].value;

      return outputs;
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
