#include <libkann/FunctionalModel.hpp>

#include <libkann/operations/ReduceOperation.hpp>

#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/copy.hpp>

#include <iterator>
#include <memory>
#include <ranges>
#include <numeric>
#include <unordered_map>

#include <cxxabi.h>

namespace kann
{
  template<typename Callback>
  static void walk(const std::shared_ptr<const OldVariable>& variable, const Callback& callback)
  {
    if(!callback(variable))
      return;

    for(auto& input : variable->inputs)
      walk(input.variable, callback);
  }

  template<typename Callback>
  static void walk(
      const std::shared_ptr<const OldVariable>& input,
      const std::shared_ptr<const OldVariable>& output,
      const std::vector<FunctionalModel::FeedBack>& feedBacks,
      const Callback& callback)
  {
    walk(output, callback);
    for(const auto& feedBack : feedBacks)
      walk(feedBack.output, callback);
  }

  FunctionalModel::FunctionalModel(std::shared_ptr<const OldVariable> input, std::shared_ptr<const OldVariable> output, std::vector<FeedBack> feedBacks)
  {
    // 1: Enumerate all varaibles in a well-defined order
    std::vector<std::shared_ptr<const OldVariable>> variables;
    {
      std::set<std::shared_ptr<const OldVariable>> set;
      walk(input, output, feedBacks, [&variables, &set](const std::shared_ptr<const OldVariable>& variable){
        if(set.contains(variable))
          return false;

        variables.push_back(variable);
        set.insert(variable);
        return true;
      });
    }

    // 2: Associate each variables with a node
    std::vector<Node> nodes(variables.size());
    for(size_t i=0; i<nodes.size(); ++i)
      nodes[i] = Node{.size = variables[i]->size};

    // 3: Associate each variables/nodes with a vertex
    std::vector<vertex_type> vertices(variables.size());
    for(size_t i=0; i<vertices.size(); ++i)
      vertices[i] = boost::add_vertex(VertexProperty{.nodeIndex = i}, m_graph);

    // 4: Establish a map between variable to index
    std::unordered_map<std::shared_ptr<const OldVariable>, size_t> indicesMap;
    for(size_t i=0; i<variables.size(); ++i)
      indicesMap.emplace(variables[i], i);

    // 5: Make appropriate connection
    for(const auto& variable : variables)
      for(const auto& input : variable->inputs)
      {
        // Given a variable, how do we find the corresponding vertex
        const size_t inputIndex  = indicesMap.at(input.variable);
        const size_t outputIndex = indicesMap.at(variable);
        const auto& layerIndex = this->addLayer(input.layer);
        boost::add_edge(vertices[inputIndex], vertices[outputIndex], EdgeProperty{.layerIndex = layerIndex}, m_graph);
      }

    // 6: Save indices of interest
    m_inputNodeIndex  = indicesMap.at(input);
    m_outputNodeIndex = indicesMap.at(output);

    for(const auto& feedBack : feedBacks)
      m_feedBacksNodeIndices.push_back(std::make_pair(
        indicesMap.at(feedBack.input),
        indicesMap.at(feedBack.output)
      ));

    // 7: Populate member variables
    m_nodes = std::move(nodes);

    // 8: Build the ordering
    this->build();
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

  void FunctionalModel::write_graphviz(std::ostream& os) const
  {
    auto vertexWriter = [](std::ostream& os, vertex_type vertex){};
    auto edgeWriter = [this](std::ostream& os, edge_type edge){
      const Layer& layer = this->layer(m_graph[edge].layerIndex);
      os << "[label=\"";
      os << demangle(typeid(layer).name()) << "\\n";
      os << "input_size=" << layer.inputSize() << "\\n";
      os << "output_size=" << layer.outputSize() << "\\n";
      os << "tag=" << layer.tag() << "\\n";
      os << "\"]";
    };
    boost::write_graphviz(os, m_graph, vertexWriter, edgeWriter);
  }

  std::unique_ptr<Layer> FunctionalModel::clone() const
  {
    return std::make_unique<FunctionalModel>(*this);
  }

  auto FunctionalModel::operator()(std::shared_ptr<const Variable> input, StateVariables state) const -> std::pair<std::shared_ptr<const Variable>, StateVariables>
  {
    std::vector<std::shared_ptr<const Variable>> variables;
    variables.resize(m_nodes.size());

    // Input variables
    variables[m_inputNodeIndex] = std::move(input);

    // State variables
    assert(state.size() == m_feedBacksNodeIndices.size());
    for(size_t i=0; i<m_feedBacksNodeIndices.size(); ++i)
    {
      const auto [inputNodeIndex, outputNodeIndex] = m_feedBacksNodeIndices[i];
      assert(!variables[inputNodeIndex]);
      variables[inputNodeIndex] = std::move(state[i]);
    }

    // Propagate
    for(vertex_type vertex : m_ordering)
    {
      std::vector<std::shared_ptr<const Variable>> outputVariables;
      for(auto [it, end] = boost::in_edges(vertex, m_graph); it != end; ++it)
      {
        const EdgeProperty& edgeProperty = m_graph[*it];
        const Layer& layer = this->layer(edgeProperty.layerIndex) ;

        vertex_type source = boost::source(*it, m_graph);
        auto inputVariable = variables[m_graph[source].nodeIndex];
        auto [outputVariable, state] = layer(inputVariable, StateVariables());
        assert(state.empty());

        outputVariables.push_back(outputVariable);
      }

      auto& variable = variables[m_graph[vertex].nodeIndex];
      switch(outputVariables.size())
      {
      case 0:
        assert(variable);
        break;
      case 1:
        variable = outputVariables.front();
        break;
      default:
        variable = std::make_shared<const Variable>(outputVariables, std::make_shared<ReduceOperation>(outputVariables.size()));
        break;
      }
    }

    // State variables
    assert(state.size() == m_feedBacksNodeIndices.size());
    for(size_t i=0; i<m_feedBacksNodeIndices.size(); ++i)
    {
      const auto [inputNodeIndex, outputNodeIndex] = m_feedBacksNodeIndices[i];
      assert(variables[outputNodeIndex]);
      state[i] = variables[outputNodeIndex];
    }

    // Output variables
    auto output = std::move(variables[m_outputNodeIndex]);

    return std::make_pair(std::move(output), std::move(state));
  }

  std::vector<std::shared_ptr<const Variable>> FunctionalModel::makeStateVariables() const
  {
    return std::vector(m_feedBacksNodeIndices.size(), std::make_shared<const Variable>());
  }

  std::vector<Tensor> FunctionalModel::makeState() const
  {
    std::vector<Tensor> result;
    for(const auto& [inputNodeIndex, outputNodeIndex] : m_feedBacksNodeIndices)
    {
      assert(node(inputNodeIndex).size == node(outputNodeIndex).size);
      const size_t size = node(inputNodeIndex).size;
      result.emplace_back(size);
      result.back().asArray().setZero();
    }
    return result;
  }

  void FunctionalModel::build()
  {
    boost::topological_sort(m_graph, std::back_inserter(m_ordering));
    std::reverse(m_ordering.begin(), m_ordering.end());
  }

  FunctionalModel::Node& FunctionalModel::node(size_t index)
  {
    return m_nodes[index];
  }

  const FunctionalModel::Node& FunctionalModel::node(size_t index) const
  {
    return m_nodes[index];
  }
}
