#include <libkann/Model.hpp>

#include <libkann/layers/IdentityLayer.hpp>

#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/copy.hpp>

#include <iterator>
#include <memory>
#include <ranges>
#include <numeric>

#include <cxxabi.h>

namespace kann
{
  Model::Model(std::shared_ptr<const Variable> input, std::shared_ptr<const Variable> output,
      std::vector<FeedBack> feedBacks)
    : m_input(std::move(input)), m_output(std::move(output)),
      m_feedBacks(std::move(feedBacks))
  {
    initialize();
  }

  template<typename Callback>
  static void walk(const std::shared_ptr<const Variable>& variable, const Callback& callback)
  {
    if(!callback(variable))
      return;

    for(auto& input : variable->inputs)
      walk(input.variable, callback);
  }

  void Model::initialize()
  {
    // Build the graph
    std::set<std::shared_ptr<const Variable>> inputVariables;
    inputVariables.insert(m_input);
    for(const auto& feedBack : m_feedBacks)
      inputVariables.insert(feedBack.input);

    // 1: Create a Node for every variable
    std::map<std::shared_ptr<const Variable>, Handle> handles;
    auto callback = [this, &inputVariables, &handles](const std::shared_ptr<const Variable>& variable){
      if(handles.find(variable) != handles.end())
        return false;

      auto node = Node{
        .variable = variable,
        .data     = Eigen::VectorXd::Zero(variable->size),
        .gradient = Eigen::RowVectorXd::Zero(variable->size)
      };
      auto handle = boost::add_vertex(std::move(node), m_graph);
      handles.insert(std::make_pair(variable, handle));

      if(inputVariables.contains(variable))
        return false;

      return true;
    };
    walk(m_output, callback);
    for(auto& feedBack : m_feedBacks)
      walk(feedBack.output, callback);

    // 2: Add appropriate connection for each variable
    for(const auto& [outputVariable, outputHandle] : handles)
    {
      for(const auto& [inputVariable, layer] : outputVariable->inputs)
      {
        auto it = handles.find(inputVariable);
        if(it == handles.end())
          continue;

        const auto& inputHandle = handles[inputVariable];

        // TODO: Figure out a new way of setting tag
        auto [_, success] = boost::add_edge(inputHandle, outputHandle, Connection{layer}, m_graph);
        assert(success);
      }
    }

    // Assign the handle to input and output
    m_inputHandle = handles.at(m_input);
    m_outputHandle = handles.at(m_output);

    // Assign feedback handles
    for(const auto& feedBack : m_feedBacks)
      m_feedBackHandles.push_back(FeedBackHandle{
        .input = handles.at(feedBack.input),
        .output = handles.at(feedBack.output)
      });

    // Build the ordering
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
      const Node& node = m_graph[handle];
      os << "[label=\"";
      os << "size=" << node.variable->size << "\\n";
      os << "\"]";
    };

    auto edgeWriter = [this](std::ostream& os, const auto& handle){
      const Connection& connection = m_graph[handle];
      const auto& layer = connection.layer;
      os << "[label=\"";
      os << demangle(typeid(*layer).name()) << "\\n";
      os << "input_size=" << layer->inputSize() << "\\n";
      os << "output_size=" << layer->outputSize() << "\\n";
      os << "tag=" << layer->tag() << "\\n";
      os << "\"]";
    };
    boost::write_graphviz(os, m_graph, vertexWriter, edgeWriter);
  }

  void Model::feedForward(Eigen::VectorXd input)
  {
    std::map<Handle, Eigen::VectorXd> feedBackData;

    /* Save feedBack data
     *
     * We kinda have a chicken-and-egg dilemma - to get the
     * input for the feedback loop we need the output of the feedback loop to
     * feed into the model first. The way to solve it is to make sure that the
     * input variable of the feedback loop is zero by default, even if no
     * feedForward has been called yet. This is handled by the fact that
     * Variable::constant() by default initializing all its member vectors to
     * zero vectors. Not doing so *MAY* lead to weird crashes from uninitialized
     * value. */
    for(const auto& feedBackHandle : m_feedBackHandles)
      feedBackData.insert(std::make_pair(feedBackHandle.input, m_graph[feedBackHandle.output].data));

    // Zero all pre-existing node data
    for(auto [begin, end] = boost::vertices(m_graph); begin != end; ++begin)
    {
      Node& node = m_graph[*begin];
      node.data.setZero();
    }

    // Restore feedBack data
    for(auto& [handle, data] : feedBackData)
      m_graph[handle].data = std::move(data);

    m_graph[m_inputHandle].data = std::move(input);
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

  double Model::cost(const Eigen::VectorXd& expectedOutput) const
  {
    return (m_graph[m_outputHandle].data - expectedOutput).squaredNorm();
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

    m_graph[m_outputHandle].gradient = 2.0 * (m_graph[m_outputHandle].data - expectedOutput);
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

  void Model::train(double learningRate, unsigned tags)
  {
    for(auto [begin, end] = boost::edges(m_graph); begin != end; ++begin)
    {
      auto& connection = m_graph[*begin];
      if(connection.layer->tag() & tags)
        connection.layer->train(learningRate, connection.layerGradient);
    }
  }

  template<typename Callback>
  static void walk2(const std::shared_ptr<const Variable>& variable1, const std::shared_ptr<const Variable>& variable2, const Callback& callback)
  {
    if(!callback(variable1, variable2))
      return;

    assert(variable1->inputs.size() == variable2->inputs.size());
    const size_t size = variable1->inputs.size();
    for(size_t i=0; i<size; ++i)
      walk2(variable1->inputs[i].variable, variable2->inputs[i].variable, callback);
  }


  Model Model::cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate)
  {
    // Mapping from original variables from lhs and rhs to new variables
    std::map<std::pair<std::shared_ptr<const Variable>, std::shared_ptr<const Variable>>, std::shared_ptr<Variable>> variables;

    // Make a deep copy of lhs and rhs variables
    {
      auto callback = [&variables](const std::shared_ptr<const Variable>& variable1, const std::shared_ptr<const Variable>& variable2) -> bool {
        if(variables.find(std::make_pair(variable1, variable2)) != variables.end())
          return false;

        assert(variable1->size == variable2->size);
        const size_t size = variable1->size;
        auto variable = Variable::constant(size);
        variables.insert(std::make_pair(std::make_pair(variable1, variable2), std::move(variable)));
        return true;
      };
      walk2(lhs.m_output, rhs.m_output, callback);

      assert(lhs.m_feedBacks.size() == rhs.m_feedBacks.size());
      const size_t size = lhs.m_feedBacks.size();
      for(size_t i=0; i<size; ++i)
        walk2(lhs.m_feedBacks[i].output, rhs.m_feedBacks[i].output, callback);
    }


    // Fix up variable inputs
    for(const auto& [key, variable] : variables)
    {
      const auto& [variable1, variable2] = key;

      assert(variable1->inputs.size() == variable2->inputs.size());
      const size_t size = variable1->inputs.size();
      variable->inputs.resize(size);
      for(size_t i=0; i<size; ++i)
      {
        const auto& input1 = variable1->inputs[i];
        const auto& input2 = variable2->inputs[i];

        auto it = variables.find(std::make_pair(input1.variable, input2.variable));
        assert(it != variables.end());
        auto childVariable = it->second;
        auto childLayer = Layer::cross(*input1.layer, *input2.layer, engine, mutationRate);

        auto& input = variable->inputs[i];

        input.variable = std::move(childVariable);
        input.layer    = std::move(childLayer);
      }
    }

    // Lookup the necessary variable and construct the model
    auto itInput = variables.find(std::make_pair(lhs.m_input, rhs.m_input));
    assert(itInput != variables.end());
    auto input = itInput->second;

    auto itOutput = variables.find(std::make_pair(lhs.m_output, rhs.m_output));
    assert(itOutput != variables.end());
    auto output = itOutput->second;

    std::vector<FeedBack> feedBacks;

    assert(lhs.m_feedBacks.size() == rhs.m_feedBacks.size());
    const size_t size = lhs.m_feedBacks.size();
    feedBacks.resize(size);
    for(size_t i=0; i<size; ++i)
    {
      auto itInput  = variables.find(std::make_pair(lhs.m_feedBacks[i].input,  rhs.m_feedBacks[i].input));
      assert(itInput != variables.end());
      auto input = itInput->second;

      auto itOutput = variables.find(std::make_pair(lhs.m_feedBacks[i].output, rhs.m_feedBacks[i].output));
      assert(itOutput != variables.end());
      auto output = itOutput->second;

      feedBacks[i].input = std::move(input);
      feedBacks[i].output = std::move(output);
    }

    return Model(std::move(input), std::move(output), std::move(feedBacks));
  }

  Model buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers, unsigned tag)
  {
    auto input = Variable::constant(layers.front()->inputSize());
    auto output = input;
    for(auto& layer : layers)
      output = output | layer;

    return Model(std::move(input), std::move(output));
  }

  Model buildSimpleRecurrentModel(std::vector<std::shared_ptr<Layer>> layers, size_t memory, unsigned tag)
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
    return Model(std::move(realInput), std::move(realOutput), {feedBack});
  }

  std::pair<Model, Model> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers)
  {
    auto input = Variable::constant(encoderLayers.front()->inputSize());
    auto middle = input;
    for(auto& layer : encoderLayers)
    {
      layer->tag(TAG_ENCODDER);
      middle = middle | layer;
    }

    auto output = middle;
    for(auto& layer : decoderLayers)
    {
      layer->tag(TAG_DECODDER);
      output = output | layer;
    }

    auto autoEncoderModel = Model(input, output);
    auto decoderModel     = Model(middle, output);
    return {autoEncoderModel, decoderModel};
  }

  std::tuple<Model, Model, Model> buildSimpleGANModel(std::vector<std::shared_ptr<Layer>> generatorLayers, std::vector<std::shared_ptr<Layer>> discriminatorLayers)
  {
    auto input = Variable::constant(generatorLayers.front()->inputSize());
    auto middle = input;
    for(auto& layer : generatorLayers)
    {
      layer->tag(TAG_GAN_GENERATOR);
      middle = middle | layer;
    }

    auto output = middle;
    for(auto& layer : discriminatorLayers)
    {
      layer->tag(TAG_GAN_DISCRIMINATOR);
      output = output | layer;
    }

    auto GANModel           = Model(input, output);
    auto generatorModel     = Model(input, middle);
    auto discriminatorModel = Model(middle, output);
    return {GANModel, generatorModel, discriminatorModel};
  }
}
