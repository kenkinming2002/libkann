#include "NeuralNetwork.hpp"

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, size_t memory) 
  : m_memory(memory), 
    m_topology(topology.size()), m_layers(topology.size()), m_weights(topology.size()-1)
{
  for(size_t i = 0; i<m_topology.size(); ++i)
    m_topology[i] = topology[i];

  m_topology.front() += memory;
  m_topology.back() += memory;

  for(size_t i = 0; i<m_layers.size(); ++i)
    m_layers[i] = m_topology[i];

  for(size_t i=0; i<m_topology.size()-1; i++)
  {
    size_t previousLayerSize = m_topology[i], nextLayerSize = m_topology[i+1];
    m_weights[i] = Eigen::MatrixXd(nextLayerSize, previousLayerSize);
  }

  m_output = Eigen::VectorXd(m_topology.back());
}

template<typename PRNG>
NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, PRNG& prng, size_t memory) 
  : m_memory(memory), m_topology(topology.size()), m_layers(topology.size()), m_weights(topology.size()-1)
{
  for(size_t i = 0; i<m_topology.size(); ++i)
    m_topology[i] = topology[i];

  m_topology.front() += memory;
  m_topology.back() += memory;

  for(size_t i = 0; i<m_layers.size(); ++i)
    m_layers[i] = m_topology[i];

  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for(size_t i=0; i<m_topology.size()-1; i++)
  {
    size_t previousLayerSize = m_topology[i], nextLayerSize = m_topology[i+1];
    m_weights[i] = Eigen::MatrixXd::NullaryExpr(nextLayerSize, previousLayerSize,[&](){
      return distribution(prng);
    });
  }

  m_output = Eigen::VectorXd(m_topology.back());
}

template NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, std::mt19937& prng, size_t memory);

void NeuralNetwork::input(std::initializer_list<double> input)
{
  for(size_t i=0; i<input.size(); ++i)
    m_layers.front().input()(i) = *(input.begin() + i);

  for(size_t i=input.size(); i<input.size()+m_memory; ++i)
    m_layers.front().input()(i) = output(i);
}

double NeuralNetwork::output(size_t i) const
{
  return m_output(i);
}

const Eigen::VectorXd& NeuralNetwork::output() const
{
  return m_output;
}

void NeuralNetwork::feedForward()
{
  for(size_t i=0; i<m_weights.size(); ++i)
    m_layers[i+1].input() = m_weights[i] * m_layers[i].output();

  m_output = m_layers.back().output();
}

template<typename PRNG>
NeuralNetwork NeuralNetwork::cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, PRNG& prng, double mutationRate)
{
  std::uniform_int_distribution<> distribution(0, 1);

  std::uniform_real_distribution<double> mutationDistribution(0.0, 1.0);
  std::uniform_real_distribution<double> weightDistribution(-1.0, 1.0);

  // Note: assertion somtimes failed bug
  if(lhs.m_weights.size() != rhs.m_weights.size())
  {
    std::cerr << lhs.m_weights.size() << " " << rhs.m_weights.size() << std::endl;
    assert(lhs.m_weights.size() == rhs.m_weights.size());
  }

  NeuralNetwork output = lhs;
  for(size_t i=0; i<lhs.m_weights.size(); ++i)
  {
    auto& lhsWeight = lhs.m_weights[i];
    auto& rhsWeight = rhs.m_weights[i]; 
    auto& outputWeight = output.m_weights[i];

    assert(lhsWeight.size() != 0);
    assert(lhsWeight.size() == rhsWeight.size());
    assert(rhsWeight.size() == outputWeight.size());

    for(long j=0; j<lhsWeight.size(); ++j)
    {
      if(distribution(prng) == 0)
        outputWeight.data()[j] = lhsWeight.data()[j];
      else
        outputWeight.data()[j] = rhsWeight.data()[j];

      if(mutationDistribution(prng) < mutationRate)
        outputWeight.data()[j] = weightDistribution(prng);
    }
  }

  return output;
}

template NeuralNetwork NeuralNetwork::cross<std::mt19937>(const NeuralNetwork& lhs, const NeuralNetwork& rhs, std::mt19937& prng, double mutationRate);

