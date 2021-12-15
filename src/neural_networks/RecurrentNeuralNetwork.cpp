#include <libkann/neural_networks/RecurrentNeuralNetwork.hpp>

namespace kann
{
  RecurrentNeuralNetwork::RecurrentNeuralNetwork(size_t memory)
    : m_memory(memory) {}

  size_t RecurrentNeuralNetwork::inputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.front()->inputSize() - m_memory;
  }

  size_t RecurrentNeuralNetwork::outputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.back()->outputSize() - m_memory;
  }

  void RecurrentNeuralNetwork::feedForward(Eigen::VectorXd input, RecurrentFeedForwardResult& result) const
  {
    if(result.memory.size() != m_memory)
      result.memory = Eigen::VectorXd::Zero(m_memory);

    Eigen::VectorXd realInput(input.size()+result.memory.size());
    realInput << input, result.memory;

    result.data.resize(m_layers.size()+1);
    result.data.front() = std::move(realInput);
    for(size_t i=0; i<m_layers.size(); ++i)
      m_layers[i]->feedForward(result.data[i], result.data[i+1]);

    result.output = result.data.back().head(result.data.back().size() - m_memory);
    result.memory = result.data.back().tail(m_memory);
  }

  void RecurrentNeuralNetwork::randomize(std::default_random_engine& engine)
  {
    for(auto& layer : m_layers)
      layer->randomize(engine);
  }

  RecurrentNeuralNetwork RecurrentNeuralNetwork::cross(const RecurrentNeuralNetwork& lhs, const RecurrentNeuralNetwork& rhs, std::default_random_engine& engine, double mutationRate)
  {
    assert(lhs.m_memory == rhs.m_memory);
    assert(lhs.m_layers.size() == rhs.m_layers.size());

    const auto memory    = lhs.m_memory;
    const auto layerSize = lhs.m_layers.size();

    RecurrentNeuralNetwork result(memory);
    for(size_t i=0; i<layerSize; ++i)
      result.addLayer(Layer::cross(*lhs.m_layers[i], *rhs.m_layers[i], engine, mutationRate));

    return result;
  }

  void RecurrentNeuralNetwork::addLayer(std::unique_ptr<Layer> layer)
  {
    m_layers.push_back(std::move(layer));
  }
}
