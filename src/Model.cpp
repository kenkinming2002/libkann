#include <libkann/Model.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>

#include <ranges>
#include <numeric>

#include <cxxabi.h>

namespace kann
{
  size_t Model::add(std::shared_ptr<Layer> layer)
  {
    m_nodes.push_back(Node{.layer = std::move(layer)});
    return m_nodes.size() - 1;
  }

  void Model::connect(size_t parentID, size_t childID)
  {
    boost::add_edge(parentID, childID, m_graph);
  }

  void Model::build(size_t inputID, size_t outputID)
  {
    m_inputID  = inputID;
    m_outputID = outputID;

    boost::topological_sort(m_graph, std::back_inserter(m_ordering));
    std::reverse(m_ordering.begin(), m_ordering.end());

    std::cout << "Ordering:";
    for(auto id : m_ordering)
      std::cout << id << " ";
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
    boost::write_graphviz(os, m_graph, [this](std::ostream& os, const auto& id){
      auto& layer = *m_nodes[id].layer.get();
      os << "[label=\"";
      os << demangle(typeid(layer).name()) << "\\n";
      os << "input_size=" << layer.inputSize() << "\\n";
      os << "output_size=" << layer.outputSize() << "\\n";
      os << "\"]";
    });
  }

  Eigen::VectorXd Model::feedForward(Eigen::VectorXd input)
  {
    m_nodes[m_inputID].input = std::move(input);

    Eigen::VectorXd finalOutput;
    for(size_t id : m_ordering)
    {
      const auto& node = m_nodes[id];

      Eigen::VectorXd output;
      node.layer->feedForward(node.input, output);
      for(auto [begin, end] = boost::out_edges(id, m_graph); begin != end; ++begin)
      {
        auto childID = boost::target(*begin, m_graph);
        auto& childNode = m_nodes[childID];
        childNode.input = output;
      }

      if(id == m_outputID)
        finalOutput = std::move(output);
    }
    return finalOutput;
  }

  void Model::backPropagate(const Eigen::VectorXd& output, const Eigen::VectorXd& expectedOutput)
  {
    m_nodes[m_outputID].outputGradient = 2.0 * (output - expectedOutput);
    for(auto it = m_ordering.rbegin(); it != m_ordering.rend(); ++it)
    {
      auto id = *it;
      auto& node = m_nodes[id];

      Eigen::RowVectorXd inputGradient;
      node.layer->backPropagate(node.input, node.outputGradient, inputGradient, node.layerGradient);

      for(auto [begin, end] = boost::in_edges(id, m_graph); begin != end; ++begin)
      {
        auto parentID = boost::source(*begin, m_graph);
        auto& parentNode = m_nodes[parentID];
        parentNode.outputGradient = inputGradient;
      }
    }
  }

  void Model::train(double learningRate)
  {
    for(auto& node : m_nodes)
      node.layer->train(learningRate, node.layerGradient);
  }

  Model buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers)
  {
    Model model;
    std::vector<size_t> ids; // Not strictly necessary, since ids are allocated incrementally

    for(auto& layer: layers)
      ids.push_back(model.add(std::move(layer)));

    for(size_t i=0; i<ids.size()-1; ++i)
      model.connect(ids[i], ids[i+1]);

    model.build(ids.front(), ids.back());

    return model;
  }
}
