#include <boost/graph/subgraph.hpp>
#include <libkann/Model.hpp>

#include <libkann/layers/IdentityLayer.hpp>

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

  Model::Model(std::shared_ptr<const Variable> input, std::shared_ptr<const Variable> output, std::vector<FeedBack> feedBacks)
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
        auto connection = Connection{.layer = input.layer};
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

  void Model::write_graphviz(std::ostream& os) const
  {
    auto vertexWriter = [this](std::ostream& os, const auto& vertex_descriptor){
      const Node& node = m_graph[vertex_descriptor];
      os << "[label=\"";
      os << "size=" << node.size << "\\n";
      os << "\"]";
    };

    auto edgeWriter = [this](std::ostream& os, const auto& edge_descriptor){
      const Connection& connection = m_graph[edge_descriptor];
      const Layer& layer = *connection.layer;
      os << "[label=\"";
      os << demangle(typeid(layer).name()) << "\\n";
      os << "input_size=" << layer.inputSize() << "\\n";
      os << "output_size=" << layer.outputSize() << "\\n";
      os << "tag=" << layer.tag() << "\\n";
      os << "\"]";
    };
    boost::write_graphviz(os, m_graph, vertexWriter, edgeWriter);
  }

  std::unique_ptr<Model> Model::cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate)
  {
    return std::unique_ptr<Model>(static_cast<Model*>(Layer::cross(lhs, rhs, engine, mutationRate).release()));
  }

  void Model::randomize(std::default_random_engine& engine)
  {
    for(auto [it, end] = boost::edges(m_graph); it != end; ++it)
      m_graph[*it].layer->randomize(engine);
  }

  void Model::train(double learningRate, unsigned tags)
  {
    for(auto [it, end] = boost::edges(m_graph); it != end; ++it)
      if(m_graph[*it].layer->tag() & tags)
        m_graph[*it].layer->train(learningRate);
      else
        m_graph[*it].layer->train(0.0); // Clear the gradient
  }

  std::unique_ptr<Layer> Model::clone() const
  {
    auto result = std::make_unique<Model>(*this);
    for(auto [it, end] = boost::edges(result->m_graph); it !=  end; ++it)
    {
      Connection& connection = result->m_graph[*it];
      connection.layer = connection.layer->clone();
    }
    return result;
  }

  Eigen::VectorXd Model::feedForward()
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

        connection.layer->input(inputNode.data);
        outputNode.data += connection.layer->feedForward();
      }
    }

    return m_graph[m_output_vertex_descriptor].data;
  }

  Eigen::VectorXd Model::backPropagate()
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

        connection.layer->outputGradient(outputNode.gradient);
        inputNode.gradient += connection.layer->backPropagate();
      }
    }
    return m_graph[m_input_vertex_descriptor].gradient;
  }

  std::vector<std::span<double>> Model::params()
  {
    std::vector<std::span<double>> params;
    for(auto [it, end] = boost::edges(m_graph); it !=  end; ++it)
    {
      Connection& connection = m_graph[*it];
      auto paramsConnection = connection.layer->params();
      params.insert(params.end(), std::move_iterator(paramsConnection.begin()), std::move_iterator(paramsConnection.end()));
    }
    return params;
  }

  std::vector<std::span<const double>> Model::params() const
  {
    std::cout << "Begin:" << std::endl;
    std::vector<std::span<const double>> params;
    for(auto [it, end] = boost::edges(m_graph); it !=  end; ++it)
    {
      const Connection& connection = m_graph[*it];
      std::cout << "Params:" << typeid(*connection.layer).name() << std::endl;
      auto paramsConnection = connection.layer->params();
      params.insert(params.end(), std::move_iterator(paramsConnection.begin()), std::move_iterator(paramsConnection.end()));
    }
    std::cout << "End:" << std::endl;
    return params;
  }


  std::vector<std::span<double>> Model::paramsGradient()
  {
    std::vector<std::span<double>> paramsGradient;
    for(auto [it, end] = boost::edges(m_graph); it !=  end; ++it)
    {
      Connection& connection = m_graph[*it];
      auto paramsGradientConnection = connection.layer->paramsGradient();
      paramsGradient.insert(paramsGradient.end(), std::move_iterator(paramsGradientConnection.begin()), std::move_iterator(paramsGradientConnection.end()));
    }
    return paramsGradient;
  }

  std::vector<std::span<const double>> Model::paramsGradient() const
  {
    std::vector<std::span<const double>> paramsGradient;
    for(auto [it, end] = boost::edges(m_graph); it !=  end; ++it)
    {
      const Connection& connection = m_graph[*it];
      auto paramsGradientConnection = connection.layer->paramsGradient();
      paramsGradient.insert(paramsGradient.end(), std::move_iterator(paramsGradientConnection.begin()), std::move_iterator(paramsGradientConnection.end()));
    }
    return paramsGradient;
  }

  std::shared_ptr<Model> buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers, unsigned tag)
  {
    auto input = Variable::constant(layers.front()->inputSize());
    auto output = input;
    for(auto& layer : layers)
      output = output | layer;

    return std::make_shared<Model>(std::move(input), std::move(output));
  }

  std::shared_ptr<Model> buildSimpleRecurrentModel(std::vector<std::shared_ptr<Layer>> layers, size_t memory, unsigned tag)
  {
    const size_t inputSize  = layers.front()->inputSize();
    const size_t outputSize = layers.back()->outputSize();

    auto realInput = Variable::constant(inputSize-memory);
    auto memoryInput = Variable::constant(memory);

    auto input1 = realInput   | std::make_shared<IdentityLayer>(inputSize - memory, inputSize, 0                 );
    auto input2 = memoryInput | std::make_shared<IdentityLayer>(memory            , inputSize, inputSize - memory);
    auto input = input1 + input2;

    auto output = input;
    for(auto& layer : layers)
      output = output | layer;

    auto realOutput   = output | std::make_shared<IdentityLayer>(outputSize, outputSize - memory, 0                  );
    auto memoryOutput = output | std::make_shared<IdentityLayer>(outputSize, memory             , outputSize - memory);

    auto feedBack = Model::FeedBack{
      .input = std::move(memoryInput),
      .output = std::move(memoryOutput)
    };
    return std::make_shared<Model>(std::move(realInput), std::move(realOutput), std::vector<Model::FeedBack>{feedBack});
  }

  std::pair<std::shared_ptr<Model>, std::shared_ptr<Model>> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers)
  {
    auto encoderModel = std::shared_ptr<Model>(buildSimpleFeedForwardModel(std::move(encoderLayers)));
    encoderModel->tag(TAG_ENCODDER);

    auto decoderModel = std::shared_ptr<Model>(buildSimpleFeedForwardModel(std::move(decoderLayers)));
    decoderModel->tag(TAG_DECODDER);

    auto input = Variable::constant(encoderModel->inputSize());
    auto middle = input | encoderModel;
    auto output = middle | decoderModel;

    auto autoEncoderModel = std::make_shared<Model>(std::move(input), std::move(output));

    return {std::move(autoEncoderModel), std::move(decoderModel)};
  }

  std::tuple<std::shared_ptr<Model>, std::shared_ptr<Model>, std::shared_ptr<Model>> buildSimpleGANModel(std::vector<std::shared_ptr<Layer>> generatorLayers, std::vector<std::shared_ptr<Layer>> discriminatorLayers)
  {
    auto generatorModel = std::shared_ptr<Model>(buildSimpleFeedForwardModel(std::move(generatorLayers)));
    generatorModel->tag(TAG_GAN_GENERATOR);

    auto discriminatorModel = std::shared_ptr<Model>(buildSimpleFeedForwardModel(std::move(discriminatorLayers)));
    discriminatorModel->tag(TAG_GAN_DISCRIMINATOR);

    auto input = Variable::constant(generatorModel->inputSize());
    auto middle = input | generatorModel;
    auto output = middle | discriminatorModel;

    auto GANModel = std::make_shared<Model>(std::move(input), std::move(output));

    return {std::move(GANModel), std::move(generatorModel), std::move(discriminatorModel)};
  }
}
