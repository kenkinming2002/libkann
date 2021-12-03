#include <libkann/NeuralNetwork.hpp>

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>

namespace kann
{
  size_t NeuralNetwork::inputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.front()->inputSize();
  }

  size_t NeuralNetwork::outputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.back()->outputSize();
  }

  void NeuralNetwork::randomize(std::default_random_engine& engine)
  {
    for(auto& layer : m_layers)
      layer->randomize(engine);
  }

  void NeuralNetwork::feedForward(Eigen::VectorXd input)
  {
    for(auto& layer : m_layers)
      input = layer->feedForward(input);

    m_output = input;
  }

  void NeuralNetwork::backPropagate(const Eigen::VectorXd& expectedOutput)
  {
    // TODO: Allow changing the cost function
    Eigen::VectorXd outputGradient = 2.0 * (m_output - expectedOutput);
    for(auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
    {
      auto& layer = *it;
      outputGradient = layer->backPropagate(outputGradient);
    }
  }

  void NeuralNetwork::train(double learningRate)
  {
    for(auto& layer: m_layers)
      layer->train(learningRate);
  }

  NeuralNetwork NeuralNetwork::cross(const NeuralNetwork& other, std::default_random_engine& engine, double mutationRate) const
  {
    NeuralNetwork result;
    for(size_t i=0; i<m_layers.size(); ++i)
      result.addLayer(m_layers[i]->cross(*other.m_layers[i], engine, mutationRate));

    return result;
  }

  void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer)
  {
    m_layers.push_back(std::move(layer));
    m_output = Eigen::VectorXd(this->outputSize());
  }

  const Eigen::VectorXd& NeuralNetwork::output() const
  {
    return m_output;
  }
}


