#include <libkann/Model.hpp>

#include <libkann/layers/IdentityLayer.hpp>

#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/copy.hpp>

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

    // Zero all node

    std::cout << "Ordering:";
    for(auto handle : m_ordering)
      std::cout << handle << " ";
    std::cout << std::endl;
    for(auto [begin, end] = boost::vertices(m_graph); begin != end; ++begin)
    {
      Node& node = m_graph[*begin];
      node.data     = Eigen::VectorXd::Zero(node.size);
      node.gradient = Eigen::RowVectorXd::Zero(node.size);
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
      os << "\"]";
    };
    boost::write_graphviz(os, m_graph, vertexWriter, edgeWriter);
  }

  void Model::feedForward(Eigen::VectorXd input)
  {
    std::map<Handle, Eigen::VectorXd> feedBackData;

    // Save feedBack data
    for(const auto& feedBack : m_feedBacks)
      feedBackData.insert(std::make_pair(feedBack.second, m_graph[feedBack.first].data));

    // Zero all pre-existing node data
    for(auto [begin, end] = boost::vertices(m_graph); begin != end; ++begin)
    {
      Node& node = m_graph[*begin];
      node.data.setZero();
    }

    // Restore feedBack data
    for(auto& [handle, data] : feedBackData)
      m_graph[handle].data = std::move(data);


    m_graph[m_input].data = std::move(input);
    for(const auto& handle : m_ordering)
    {
      const auto& inputNode = m_graph[handle];
      for(auto [begin, end] = boost::out_edges(handle, m_graph); begin != end; ++begin)
      {
        const auto& connection = m_graph[*begin];
        auto& outputNode = m_graph[boost::target(*begin, m_graph)];

        Eigen::VectorXd output;
        connection.layer->feedForward(inputNode.data, output);

        outputNode.data += output;
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
        connection.layer->backPropagate(inputNode.data, outputNode.gradient, inputGradient, connection.layerGradient);

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

  Model Model::cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate)
  {
    assert(boost::num_vertices(lhs.m_graph) == boost::num_vertices(rhs.m_graph));
    assert(boost::num_edges(lhs.m_graph) == boost::num_edges(rhs.m_graph));
    assert(lhs.m_input == rhs.m_input);
    assert(lhs.m_output == rhs.m_output);

    auto result = lhs;

    for(auto [begin, end] = boost::edges(result.m_graph); begin != end; ++begin)
    {
      // Hopefully the edge descriptor for result.m_graph also works for
      // rhs.m_graph
      auto& connection          = result.m_graph[*begin];
      const auto& rhsConnection = rhs.m_graph[*begin];
      connection.layer = Layer::cross(*connection.layer, *rhsConnection.layer, engine, mutationRate);
    }

    // TODO: Zero all the data

    return result;
  }

  Model buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers)
  {
    Model::Graph graph;
    for(size_t i=0; i<layers.size(); ++i)
      boost::add_vertex(Model::Node{.size = layers[i]->inputSize()}, graph);

    boost::add_vertex(Model::Node{.size = layers.back()->outputSize()}, graph);

    for(size_t i=0; i<layers.size(); ++i)
      boost::add_edge(i, i+1, Model::Connection{.layer = std::move(layers[i])}, graph);

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
      boost::add_edge(i, i+1, Model::Connection{.layer = std::move(layers[i])}, graph);

    const auto input        = boost::add_vertex(Model::Node{.size = inputSize  - memory}, graph);
    const auto output       = boost::add_vertex(Model::Node{.size = outputSize - memory}, graph);
    const auto inputMemory  = boost::add_vertex(Model::Node{.size = memory             }, graph);
    const auto outputMemory = boost::add_vertex(Model::Node{.size = memory             }, graph);

    boost::add_edge(input,       0, Model::Connection{.layer = std::make_shared<IdentityLayer>(inputSize - memory, inputSize, 0                 )}, graph);
    boost::add_edge(inputMemory, 0, Model::Connection{.layer = std::make_shared<IdentityLayer>(memory            , inputSize, inputSize - memory)}, graph);

    boost::add_edge(layers.size(), output,       Model::Connection{.layer = std::make_shared<IdentityLayer>(outputSize, outputSize - memory, 0                  )}, graph);
    boost::add_edge(layers.size(), outputMemory, Model::Connection{.layer = std::make_shared<IdentityLayer>(outputSize, memory             , outputSize - memory)}, graph);

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
