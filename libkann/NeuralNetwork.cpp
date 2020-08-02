#include "NeuralNetwork.hpp"

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>
#include <algorithm>

NeuralNetwork::NeuralNetwork(dynarray<size_t> topology) 
  : m_topology(std::move(topology)), m_layers(m_topology.size()), m_connections(m_topology.size()-1)
{
  for(size_t i = 0; i<m_layers.size(); ++i)
    m_layers[i] = Layer(m_topology[i]);

  for(size_t i=0; i<m_topology.size()-1; i++)
    m_connections[i] = Connection(m_topology[i], m_topology[i+1]);

  m_output = Eigen::VectorXd(m_topology.back());
}

template<typename PRNG>
NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, PRNG& prng) : NeuralNetwork(std::move(topology))
{
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  std::for_each(m_connections.begin(), m_connections.end(), std::bind(&Connection::randomize<PRNG>, std::placeholders::_1, prng));
}
template NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, std::mt19937& prng);

void NeuralNetwork::feedForward()
{
  for(size_t i=0; i<m_connections.size(); ++i)
  {
    Layer& inputLayer  = m_layers[i];
    Layer& outputLayer = m_layers[i+1];
    Connection& connection  = m_connections[i];

    inputLayer.feedForward();
    connection.feedForward(inputLayer, outputLayer);
  }
  m_layers.back().feedForward();
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

    Eigen::VectorXd outputLayerInputGradient = outputLayer.backPropagate(outputGradient);
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

