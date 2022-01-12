#include <libkann/FunctionalModel.hpp>

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
  static void walk(const std::shared_ptr<const Variable>& variable, const Callback& callback)
  {
    if(!callback(variable))
      return;

    for(auto& input : variable->inputs)
      walk(input.variable, callback);
  }

  FunctionalModel::FunctionalModel(std::shared_ptr<const Variable> input, std::shared_ptr<const Variable> output, std::vector<FeedBack> feedBacks)
  {
    assert(boost::num_vertices(m_graph) == 0);
    assert(boost::num_edges(m_graph) == 0);

    // 2: Associate each variable with a vertex descriptor
    std::unordered_map<std::shared_ptr<const Variable>, vertex_descriptor_type> vertex_descriptors;
    auto callbackInsert = [this, &vertex_descriptors](const std::shared_ptr<const Variable>& variable){
      auto it = vertex_descriptors.find(variable);
      if(it != vertex_descriptors.end())
        return false;

      auto node = Node{
        .size     = variable->size,
        .data     = Eigen::VectorXd::Zero(variable->size),
        .gradient = Eigen::RowVectorXd::Zero(variable->size)
      };

      auto vertex_descriptor = boost::add_vertex(std::move(node), m_graph);
      vertex_descriptors.insert({variable, vertex_descriptor});
      return true;
    };
    walk(output, callbackInsert);
    for(auto& feedBack : feedBacks)
      walk(feedBack.output, callbackInsert);

    // 3: Build the graph
    std::set<std::shared_ptr<const Variable>> set;
    auto callbackConnect = [this, &vertex_descriptors, &set](const std::shared_ptr<const Variable>& variable){
      auto it = set.find(variable);
      if(it != set.end())
        return false;

      auto output_vertex_descriptor = vertex_descriptors.at(variable);
      for(const auto& input : variable->inputs)
      {
        auto input_vertex_descriptor  = vertex_descriptors.at(input.variable);
        auto connection = Connection{.layerIndex = this->addLayer(input.layer)};
        boost::add_edge(input_vertex_descriptor, output_vertex_descriptor, std::move(connection), m_graph);
      }

      set.insert(variable);
      return true;
    };
    walk(output, callbackConnect);
    for(auto& feedBack : feedBacks)
      walk(feedBack.output, callbackConnect);

    // 4: Store vertex_descriptors of importance as member variables
    m_input_vertex_descriptor  = vertex_descriptors.at(input);
    m_output_vertex_descriptor = vertex_descriptors.at(output);

    for(const auto& feedBack : feedBacks)
      m_feedBacks_vertex_descriptors.push_back(FeedBackVertexDescriptors{
        .input_vertex_descriptor  = vertex_descriptors.at(feedBack.input),
        .output_vertex_descriptor = vertex_descriptors.at(feedBack.output)
      });

    // 5: Build the ordering
    boost::topological_sort(m_graph, std::back_inserter(m_ordering));
    std::reverse(m_ordering.begin(), m_ordering.end());

    std::cout << "Ordering:";
    for(auto handle : m_ordering)
      std::cout << handle << " ";
    std::cout << std::endl;
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
    auto vertexWriter = [this](std::ostream& os, const auto& vertex_descriptor){
      const Node& node = m_graph[vertex_descriptor];
      os << "[label=\"";
      os << "size=" << node.size << "\\n";
      os << "\"]";
    };

    auto edgeWriter = [this](std::ostream& os, const auto& edge_descriptor){
      const Connection& connection = m_graph[edge_descriptor];
      const Layer& layer = this->layer(connection.layerIndex);
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

  Eigen::VectorXd FunctionalModel::feedForward()
  {
    std::map<vertex_descriptor_type, Eigen::VectorXd> feedBackData;

    // Save feedBack data
    for(const auto& [input_vertex_descriptor, output_vertex_descriptor] : m_feedBacks_vertex_descriptors)
      feedBackData.insert({input_vertex_descriptor, m_graph[output_vertex_descriptor].data});

    // Zero all pre-existing node data
    for(auto [begin, end] = boost::vertices(m_graph); begin != end; ++begin)
    {
      Node& node = m_graph[*begin];
      node.data.setZero();
    }

    // Restore feedBack data
    for(auto& [input_vertex_descriptor, data] : feedBackData)
      m_graph[input_vertex_descriptor].data = std::move(data);

    m_graph[m_input_vertex_descriptor].data = input();

    for(vertex_descriptor_type vertex_descriptor : m_ordering)
    {
      const auto& inputNode = m_graph[vertex_descriptor];
      for(auto [begin, end] = boost::out_edges(vertex_descriptor, m_graph); begin != end; ++begin)
      {
        const auto& connection = m_graph[*begin];
        auto& outputNode = m_graph[boost::target(*begin, m_graph)];
        auto& layer = this->layer(connection.layerIndex);

        layer.input(inputNode.data);
        outputNode.data += layer.feedForward();
      }
    }

    return m_graph[m_output_vertex_descriptor].data;
  }

  Eigen::VectorXd FunctionalModel::backPropagate()
  {
    // TODO: Back propagation through time

    // Zero all pre-existing node gradients
    for(auto [begin, end] = boost::vertices(m_graph); begin != end; ++begin)
    {
      Node& node = m_graph[*begin];
      node.gradient.setZero();
    }

    m_graph[m_output_vertex_descriptor].gradient = outputGradient();
    for(auto it = m_ordering.rbegin(); it != m_ordering.rend(); ++it)
    {
      vertex_descriptor_type vertex_descriptor = *it;

      const auto& outputNode = m_graph[vertex_descriptor];
      for(auto [begin, end] = boost::in_edges(vertex_descriptor, m_graph); begin != end; ++begin)
      {
        auto& connection = m_graph[*begin];
        auto& inputNode = m_graph[boost::source(*begin, m_graph)];
        auto& layer = this->layer(connection.layerIndex);

        layer.outputGradient(outputNode.gradient);
        inputNode.gradient += layer.backPropagate();
      }
    }
    return m_graph[m_input_vertex_descriptor].gradient;
  }
}
