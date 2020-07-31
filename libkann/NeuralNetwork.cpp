#include "NeuralNetwork.hpp"

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>
#include <algorithm>

NeuralNetwork::NeuralNetwork(dynarray<size_t> topology) 
  : m_topology(std::move(topology)), m_layers(m_topology.size()), m_weights(m_topology.size()-1)
{
  for(size_t i = 0; i<m_layers.size(); ++i)
    m_layers[i] = Layer(m_topology[i]);

  for(size_t i=0; i<m_topology.size()-1; i++)
    m_weights[i] = Eigen::MatrixXd(m_topology[i+1], m_topology[i]);

  m_output = Eigen::VectorXd(m_topology.back());
}

template<typename PRNG>
NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, PRNG& prng) : NeuralNetwork(std::move(topology))
{
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for(auto& weight : m_weights)
    weight = Eigen::MatrixXd::NullaryExpr(weight.rows(), weight.cols(),[&](){ return distribution(prng); });
}
template NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, std::mt19937& prng);

void NeuralNetwork::input(size_t i, double v)
{
  m_layers.front().input()(i) = v;
}

double NeuralNetwork::output(size_t i) const
{
  return m_output(i);
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

