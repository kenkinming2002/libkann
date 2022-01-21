#include <libkann/FunctionalModel.hpp>

#include <libkann/operations/ReduceOperation.hpp>

#include <libkann/serialization/Graph.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <cereal/specialize.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>

#include <iterator>
#include <memory>
#include <ranges>
#include <numeric>
#include <unordered_map>

#include <cxxabi.h>

namespace kann
{
  class FunctionalModel : public Model
  {
  public:
    FunctionalModel() = default;
    FunctionalModel(std::shared_ptr<const FunctionalVariable> input, std::shared_ptr<const FunctionalVariable> output, std::vector<FeedBack> feedBacks);

  public:
    void write_graphviz(std::ostream& os) const override;

  public:
    std::unique_ptr<Layer> clone() const override;

  public:
    size_t inputSize()  const override { return node(m_inputNodeIndex).size; }
    size_t outputSize() const override { return node(m_outputNodeIndex).size; }

  public:
    std::vector<std::shared_ptr<Parameter>> makeStates() const override;

  public:
    std::pair<std::shared_ptr<const Variable>, StateVariables> operator()(std::shared_ptr<const Variable> input, StateVariables state) const override;

  private:
    void build();

  public:
    template<typename Archive>
    void save(Archive& archive) const
    {
      archive(cereal::base_class<Model>(this));

      archive(m_nodes);
      archive(m_inputNodeIndex, m_outputNodeIndex);
      archive(m_feedBacksNodeIndices);

      archive(GraphOutputSerializer(m_graph));
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      archive(cereal::base_class<Model>(this));

      archive(m_nodes);
      archive(m_inputNodeIndex, m_outputNodeIndex);
      archive(m_feedBacksNodeIndices);

      archive(GraphInputSerializer(m_graph));
      build();
    }

  // Node
  private:
    struct Node
    {
      size_t size;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(size);
      }
    };

  private:
    Node& node(size_t index);
    const Node& node(size_t index) const;

  private:
    std::vector<Node> m_nodes;

  private:
    size_t m_inputNodeIndex, m_outputNodeIndex;
    std::vector<std::pair<size_t, size_t>> m_feedBacksNodeIndices;

  // Graph
  private:
    struct VertexProperty
    {
      size_t nodeIndex;
      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(nodeIndex);
      }
    };

    struct EdgeProperty
    {
      size_t layerIndex;
      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(layerIndex);
      }
    };

    typedef boost::adjacency_list<
      boost::vecS, boost::vecS,
      boost::bidirectionalS,
      VertexProperty, EdgeProperty
    > graph_type;
    typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_type;
    typedef boost::graph_traits<graph_type>::edge_descriptor edge_type;

  private:
    graph_type m_graph;
    std::vector<vertex_type> m_ordering;
  };

  std::shared_ptr<Model> makeFunctionalModel(std::shared_ptr<const FunctionalVariable> input, std::shared_ptr<const FunctionalVariable> output, std::vector<FeedBack> feedBacks)
  {
    return std::make_shared<FunctionalModel>(std::move(input), std::move(output), std::move(feedBacks));
  }
}


CEREAL_REGISTER_TYPE(kann::FunctionalModel);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Model, kann::FunctionalModel);

namespace cereal
{
  template<typename Archive>
  struct specialize<Archive, kann::FunctionalModel, cereal::specialization::member_load_save> {};
}

namespace kann
{
  template<typename Callback>
  static void walk(const std::shared_ptr<const FunctionalVariable>& variable, const Callback& callback)
  {
    if(!callback(variable))
      return;

    for(auto& input : variable->inputs)
      walk(input.variable, callback);
  }

  template<typename Callback>
  static void walk(
      const std::shared_ptr<const FunctionalVariable>& input,
      const std::shared_ptr<const FunctionalVariable>& output,
      const std::vector<FeedBack>& feedBacks,
      const Callback& callback)
  {
    walk(output, callback);
    for(const auto& feedBack : feedBacks)
      walk(feedBack.output, callback);
  }

  FunctionalModel::FunctionalModel(std::shared_ptr<const FunctionalVariable> input, std::shared_ptr<const FunctionalVariable> output, std::vector<FeedBack> feedBacks)
  {
    // 1: Enumerate all varaibles in a well-defined order
    std::vector<std::shared_ptr<const FunctionalVariable>> variables;
    {
      std::set<std::shared_ptr<const FunctionalVariable>> set;
      walk(input, output, feedBacks, [&variables, &set](const std::shared_ptr<const FunctionalVariable>& variable){
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
    std::unordered_map<std::shared_ptr<const FunctionalVariable>, size_t> indicesMap;
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

  std::vector<std::shared_ptr<Parameter>> FunctionalModel::makeStates() const
  {
    std::vector<std::shared_ptr<Parameter>> result;
    result.reserve(m_feedBacksNodeIndices.size());
    for(const auto& [inputNodeIndex, outputNodeIndex] : m_feedBacksNodeIndices)
    {
      assert(node(inputNodeIndex).size == node(outputNodeIndex).size);
      result.push_back(std::make_shared<Parameter>(node(inputNodeIndex).size));
    }
    return result;
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
