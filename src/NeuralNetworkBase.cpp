#include <libkann/NeuralNetworkBase.hpp>

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>

namespace kann
{
  size_t NeuralNetworkBase::inputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.front()->inputSize();
  }

  size_t NeuralNetworkBase::outputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.back()->outputSize();
  }

  void NeuralNetworkBase::randomize(std::default_random_engine& engine)
  {
    for(auto& layer : m_layers)
      layer->randomize(engine);
  }

  Eigen::VectorXd NeuralNetworkBase::feedForward(Eigen::VectorXd input)
  {
    for(auto& layer : m_layers)
    {
      layer->input(std::move(input));
      input = layer->feedForward();
    }

    return input;
  }

  Eigen::RowVectorXd NeuralNetworkBase::backPropagate(const Eigen::RowVectorXd& outputGradient)
  {
    // TODO: Allow changing the cost function
    Eigen::RowVectorXd _outputGradient = outputGradient;
    for(auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
    {
      auto& layer = *it;
      _outputGradient = layer->backPropagate(_outputGradient);
    }
    return _outputGradient;
  }

  void NeuralNetworkBase::train(double learningRate)
  {
    for(auto& layer: m_layers)
      layer->train(learningRate);
  }

  NeuralNetworkBase NeuralNetworkBase::cross(const NeuralNetworkBase& other, std::default_random_engine& engine, double mutationRate) const
  {
    NeuralNetworkBase result;
    for(size_t i=0; i<m_layers.size(); ++i)
      result.addLayer(m_layers[i]->cross(*other.m_layers[i], engine, mutationRate));

    return result;
  }

  void NeuralNetworkBase::addLayer(std::unique_ptr<Layer> layer)
  {
    m_layers.push_back(std::move(layer));
  }
}


