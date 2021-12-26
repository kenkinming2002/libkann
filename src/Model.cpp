#include <libkann/Model.hpp>

#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>

#include <iterator>
#include <ranges>
#include <numeric>

#include <cxxabi.h>

namespace kann
{
  Model::Handle Model::addNode(size_t size)
  {
    return boost::add_vertex(Node{
      .size     = size,
      .data     = Eigen::VectorXd::Zero(size),
      .gradient = Eigen::RowVectorXd::Zero(size)
    }, m_graph);
  }

  void Model::addConnection(Handle parent, Handle child, std::shared_ptr<Layer> layer)
  {
    boost::add_edge(parent, child, Connection{
      .layer = std::move(layer),
    }, m_graph);
  }

  void Model::build(Handle input, Handle output)
  {
    m_input  = input;
    m_output = output;

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

  void Model::write_graphviz(std::ostream& os) const
  {
    auto vertexWriter = [this](std::ostream& os, const auto& handle){
      const auto& node = m_graph[handle];
      os << "[label=\"";
      os << "size=" << node.size << "\\n";
      os << "\"]";
    };

    auto edgeWriter = [this](std::ostream& os, const auto& handle){
      const auto& connection = m_graph[handle];
      const auto& layer = connection.layer;
      os << "[label=\"";
      os << demangle(typeid(*layer).name()) << "\\n";
      os << "input_size=" << layer->inputSize() << "\\n";
      os << "output_size=" << layer->outputSize() << "\\n";
      os << "\"]";
    };
    boost::write_graphviz(os, m_graph, vertexWriter, edgeWriter);
  }

  void Model::feedForward(Eigen::VectorXd input)
  {
    m_graph[m_input].data = std::move(input);
    for(const auto& handle : m_ordering)
    {
      const auto& inputNode = m_graph[handle];
      for(auto [begin, end] = boost::out_edges(handle, m_graph); begin != end; ++begin)
      {
        const auto& connection = m_graph[*begin];
        auto& outputNode = m_graph[boost::target(*begin, m_graph)];
        connection.layer->feedForward(inputNode.data, outputNode.data);
      }
    }
  }

  void Model::backPropagate(const Eigen::VectorXd& expectedOutput)
  {
    m_graph[m_output].gradient = 2.0 * (m_graph[m_output].data - expectedOutput);
    for(auto it = m_ordering.rbegin(); it != m_ordering.rend(); ++it)
    {
      const auto& handle = *it;

      const auto& outputNode = m_graph[handle];
      for(auto [begin, end] = boost::in_edges(handle, m_graph); begin != end; ++begin)
      {
        auto& connection = m_graph[*begin];
        auto& inputNode = m_graph[boost::source(*begin, m_graph)];
        connection.layer->backPropagate(inputNode.data, outputNode.gradient, inputNode.gradient, connection.layerGradient);
      }
    }
  }

  void Model::train(double learningRate)
  {
    for(auto [begin, end] = boost::edges(m_graph); begin != end; ++begin)
    {
      auto& connection = m_graph[*begin];
      connection.layer->train(learningRate, connection.layerGradient);
    }
  }

  Eigen::VectorXd Model::output() const
  {
    return m_graph[m_output].data;
  }

  Model buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers)
  {
    Model model;
    std::vector<Model::Handle> handles;
    for(size_t i=0; i<layers.size(); ++i)
      handles.push_back(model.addNode(layers[i]->inputSize()));

    // The last node is not input to any layer
    handles.push_back(model.addNode(layers.back()->outputSize()));

    for(size_t i=0; i<layers.size(); ++i)
      model.addConnection(handles[i], handles[i+1], std::move(layers[i]));

    model.build(handles.front(), handles.back());

    return model;
  }

  std::pair<Model, Model> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers)
  {
    std::vector<std::shared_ptr<Layer>> layers;
    std::copy(encoderLayers.begin(), encoderLayers.end(), std::back_inserter(layers));
    std::copy(decoderLayers.begin(), decoderLayers.end(), std::back_inserter(layers));

    auto autoEncoderModel = buildSimpleFeedForwardModel(layers);
    auto decoderModel = buildSimpleFeedForwardModel(decoderLayers);
    return {autoEncoderModel, decoderModel};
  }
}
