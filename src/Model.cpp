#include <libkann/Model.hpp>

#include <libkann/layers/IdentityLayer.hpp>

#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>

#include <iterator>
#include <ranges>
#include <numeric>

#include <cxxabi.h>

namespace kann
{
  /* Note: There actually is no guarantee(at least in the official
   *       documentation) that vertex desciptor(a.k.a Handle) stay valid after
   *       moving graph, but that should be the case, since in the case of using
   *       boost::vecS, i.e. using std::vector to store vertex, vertex desciptor
   *       is no more than a size_t which index into the std::vector of vertex */
  Model::Model(Graph graph, Handle input, Handle output, std::vector<FeedBack> feedBacks)
    : m_graph(graph), m_input(input), m_output(output), m_feedBacks(std::move(feedBacks))
  {
    boost::topological_sort(m_graph, std::back_inserter(m_ordering));
    std::reverse(m_ordering.begin(), m_ordering.end());

    std::cout << "Ordering:";
    for(auto handle : m_ordering)
      std::cout << handle << " ";
    std::cout << std::endl;

    // Compute connection.offset
    for(auto [begin, end] = boost::vertices(m_graph); begin != end; ++begin)
    {
      Node& node = m_graph[*begin];
      node.data     = Eigen::VectorXd::Zero(node.size);
      node.gradient = Eigen::RowVectorXd::Zero(node.size);

      const size_t inDegree = boost::in_degree(*begin, m_graph);

      std::vector<size_t> outputSizes(inDegree);
      for(auto [_begin, _end] = boost::in_edges(*begin, m_graph); _begin != _end; ++ _begin)
      {
        Connection& connection = m_graph[*_begin];
        outputSizes[connection.id] = connection.layer->outputSize();
      }

      std::vector<size_t> offsets(inDegree);
      std::partial_sum(outputSizes.begin(), outputSizes.end(), offsets.begin());

      for(auto [_begin, _end] = boost::in_edges(*begin, m_graph); _begin != _end; ++ _begin)
      {
        Connection& connection = m_graph[*_begin];
        connection.offset = connection.id != 0 ? offsets[connection.id-1] : 0;
      }
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
      os << "offset=" << connection.offset << "\\n";
      os << "\"]";
    };
    boost::write_graphviz(os, m_graph, vertexWriter, edgeWriter);
  }

  void Model::feedForward(Eigen::VectorXd input)
  {
    // Feedback connection
    for(const auto& feedBack : m_feedBacks)
      m_graph[feedBack.second].data = m_graph[feedBack.first].data;

    m_graph[m_input].data = std::move(input);
    for(const auto& handle : m_ordering)
    {
      const auto& inputNode = m_graph[handle];
      for(auto [begin, end] = boost::out_edges(handle, m_graph); begin != end; ++begin)
      {
        const auto& connection = m_graph[*begin];
        auto& outputNode = m_graph[boost::target(*begin, m_graph)];

        Eigen::VectorXd input = inputNode.data;
        Eigen::VectorXd output;

        connection.layer->feedForward(input, output);

        outputNode.data.segment(connection.offset, connection.layer->outputSize()) = output;
      }
    }
  }

  void Model::backPropagate(const Eigen::VectorXd& expectedOutput)
  {
    // TODO: Back propagation through time

    // Zero all pre-existing node gradients
    for(auto [begin, end] = boost::vertices(m_graph); begin != end; ++begin)
    {
      Node& node = m_graph[*begin];
      node.gradient.setZero();
    }

    m_graph[m_output].gradient = 2.0 * (m_graph[m_output].data - expectedOutput);
    for(auto it = m_ordering.rbegin(); it != m_ordering.rend(); ++it)
    {
      const auto& handle = *it;

      const auto& outputNode = m_graph[handle];
      for(auto [begin, end] = boost::in_edges(handle, m_graph); begin != end; ++begin)
      {
        auto& connection = m_graph[*begin];
        auto& inputNode = m_graph[boost::source(*begin, m_graph)];

        Eigen::RowVectorXd inputGradient;
        Eigen::RowVectorXd outputGradient = outputNode.gradient.segment(connection.offset, connection.layer->outputSize());

        connection.layer->backPropagate(inputNode.data, outputGradient, inputGradient, connection.layerGradient);

        inputNode.gradient += inputGradient;
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
    Model::Graph graph;
    for(size_t i=0; i<layers.size(); ++i)
      boost::add_vertex(Model::Node{.size = layers[i]->inputSize()}, graph);

    boost::add_vertex(Model::Node{.size = layers.back()->outputSize()}, graph);

    for(size_t i=0; i<layers.size(); ++i)
      boost::add_edge(i, i+1, Model::Connection{.id = 0, .layer = std::move(layers[i])}, graph);

    return Model(graph, 0, layers.size());
  }

  Model buildSimpleRecurrentModel(std::vector<std::shared_ptr<Layer>> layers, size_t memory)
  {
    const size_t inputSize  = layers.front()->inputSize();
    const size_t outputSize = layers.back()->outputSize();

    Model::Graph graph;
    for(size_t i=0; i<layers.size(); ++i)
      boost::add_vertex(Model::Node{.size = layers[i]->inputSize()}, graph);

    boost::add_vertex(Model::Node{.size = layers.back()->outputSize()}, graph);

    for(size_t i=0; i<layers.size(); ++i)
      boost::add_edge(i, i+1, Model::Connection{.id = 0, .layer = std::move(layers[i])}, graph);

    const auto input        = boost::add_vertex(Model::Node{.size = inputSize  - memory}, graph);
    const auto output       = boost::add_vertex(Model::Node{.size = outputSize - memory}, graph);
    const auto inputMemory  = boost::add_vertex(Model::Node{.size = memory             }, graph);
    const auto outputMemory = boost::add_vertex(Model::Node{.size = memory             }, graph);

    boost::add_edge(input,       0, Model::Connection{.id = 0, .layer = std::make_shared<IdentityLayer>(inputSize - memory, inputSize - memory, 0)}, graph);
    boost::add_edge(inputMemory, 0, Model::Connection{.id = 1, .layer = std::make_shared<IdentityLayer>(memory            , memory            , 0)}, graph);

    boost::add_edge(layers.size(), output,       Model::Connection{.id = 0, .layer = std::make_shared<IdentityLayer>(outputSize, outputSize - memory, 0)}, graph);
    boost::add_edge(layers.size(), outputMemory, Model::Connection{.id = 0, .layer = std::make_shared<IdentityLayer>(outputSize, memory             , outputSize - memory)}, graph);

    auto feedBack = Model::FeedBack{.first = outputMemory, .second = inputMemory};
    return Model(graph, input, output, {feedBack});
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
