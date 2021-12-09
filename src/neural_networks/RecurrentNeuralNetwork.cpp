#include <libkann/neural_networks/RecurrentNeuralNetwork.hpp>

namespace kann
{
  RecurrentNeuralNetwork::RecurrentNeuralNetwork(size_t memory)
    : m_memory(Eigen::VectorXd::Zero(memory)) {}

  size_t RecurrentNeuralNetwork::inputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.front()->inputSize() - m_memory.size();
  }

  size_t RecurrentNeuralNetwork::outputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.back()->outputSize() - m_memory.size();
  }

  void RecurrentNeuralNetwork::randomize(std::default_random_engine& engine)
  {
    for(auto& layer : m_layers)
      layer->randomize(engine);
  }

  void RecurrentNeuralNetwork::feedForward(Eigen::VectorXd input)
  {
    Eigen::VectorXd realInput(input.size()+m_memory.size());
    realInput << input, m_memory;

    for(auto& layer : m_layers)
    {
      layer->input(std::move(realInput));
      realInput = layer->feedForward();
    }

    Eigen::VectorXd realOutput = std::move(realInput);
    m_output = realOutput.head(realOutput.size() - m_memory.size());
    m_memory = realOutput.tail(m_memory.size());
  }

  RecurrentNeuralNetwork RecurrentNeuralNetwork::cross(const RecurrentNeuralNetwork& other, std::default_random_engine& engine, double mutationRate) const
  {
    RecurrentNeuralNetwork result(m_memory.size());
    for(size_t i=0; i<m_layers.size(); ++i)
      result.addLayer(m_layers[i]->cross(*other.m_layers[i], engine, mutationRate));

    return result;
  }

  void RecurrentNeuralNetwork::addLayer(std::unique_ptr<Layer> layer)
  {
    m_layers.push_back(std::move(layer));
  }

  const Eigen::VectorXd& RecurrentNeuralNetwork::output() const
  {
    return m_output;
  }
}
