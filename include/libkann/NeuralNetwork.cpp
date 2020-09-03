#include "NeuralNetwork.hpp"

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>

NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, ActivationFunction activationFunction) 
  : m_layers(topology.size()), m_connections(topology.size()-1), m_activationFunction(activationFunction)
{
  for(size_t i = 0; i<m_layers.size(); ++i)
    m_layers[i] = Layer(topology[i]);

  for(size_t i=0; i<topology.size()-1; i++)
    m_connections[i] = Connection(topology[i], topology[i+1]);

  m_output = Eigen::VectorXd(topology.back());
}

NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, dynarray<Eigen::MatrixXd> weights, ActivationFunction activationFunction)
  : NeuralNetwork(topology, activationFunction)
{
  // TODO: optimize this to prevent redundant allocation
  for(size_t i=0; i<m_connections.size(); ++i)
    m_connections[i] = Connection(std::move(weights[i]));
}

template<typename PRNG>
NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, PRNG& prng, ActivationFunction activationFunction) 
  : NeuralNetwork(std::move(topology), activationFunction)
{
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  std::for_each(m_connections.begin(), m_connections.end(), std::bind(&Connection::randomize<PRNG>, std::placeholders::_1, std::ref(prng)));
}
template NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, std::mt19937& prng, ActivationFunction activationFunction);

void NeuralNetwork::feedForward()
{
  for(size_t i=0; i<m_connections.size(); ++i)
  {
    Layer& inputLayer  = m_layers[i];
    Layer& outputLayer = m_layers[i+1];
    Connection& connection  = m_connections[i];

    inputLayer.feedForward(m_activationFunction);
    connection.feedForward(inputLayer, outputLayer);
  }
  m_layers.back().feedForward(m_activationFunction);
  m_output = m_layers.back().output();
}

void NeuralNetwork::backPropagate(const Eigen::VectorXd& expectedOutput)
{
  Eigen::VectorXd outputGradient = 2.0 * (m_output - expectedOutput); 

  for(size_t i=m_layers.size()-1; i>0; --i)
  {
    Layer& outputLayer = m_layers[i];
    Layer& inputLayer  = m_layers[i-1];
    Connection& connection  = m_connections[i-1];

    Eigen::VectorXd outputLayerInputGradient = outputLayer.backPropagate(outputGradient, m_activationFunction);
    Eigen::VectorXd inputLayerOutputGradient = connection.backPropagate(inputLayer.output(), outputLayerInputGradient);

    outputGradient = inputLayerOutputGradient;
  }
}

void NeuralNetwork::train(double learningRate)
{
  for(auto& connection: m_connections)
    connection.train(learningRate);
}

template<typename PRNG>
NeuralNetwork NeuralNetwork::cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, PRNG& prng, double mutationRate)
{
  NeuralNetwork result = lhs;
  for(size_t i=0; i<lhs.m_connections.size(); ++i)
    result.m_connections[i] = Connection::cross(lhs.m_connections[i], rhs.m_connections[i], prng, mutationRate);
  return result;
}

template NeuralNetwork NeuralNetwork::cross<std::mt19937>(const NeuralNetwork& lhs, const NeuralNetwork& rhs, std::mt19937& prng, double mutationRate);

