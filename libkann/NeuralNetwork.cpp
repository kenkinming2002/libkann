#include "NeuralNetwork.hpp"

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>

static Eigen::VectorXd convert(const std::vector<double>& input)
{
  Eigen::VectorXd output(input.size());
  for(size_t i=0; i<input.size(); ++i)
    output(i) = input[i];
  return output;
}

static std::vector<double> convert(Eigen::VectorXd input)
{
  return std::vector<double>(input.data(), input.data() + input.size());
}

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, size_t memory) : m_memory(memory), m_topology(topology)
{
  m_topology.front() += memory;
  m_topology.back() += memory;

  // Layers
  for(size_t size: m_topology)
    m_layers.emplace_back(size);

  m_output = Eigen::VectorXd(m_topology.back());

  for(size_t i=0; i<m_topology.size()-1; i++)
  {
    size_t a = m_topology[i], b = m_topology[i+1];
    m_weights.emplace_back(Eigen::MatrixXd(b, a));
  }
}

template<typename PRNG>
NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, PRNG& prng, size_t memory) : m_memory(memory), m_topology(topology)
{
  m_topology.front() += memory;
  m_topology.back() += memory;

  // Layers
  for(size_t size: m_topology)
    m_layers.emplace_back(size);

  m_output = Eigen::VectorXd(m_topology.back());

  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  for(size_t i=0; i<m_topology.size()-1; i++)
  {
    size_t a = m_topology[i], b = m_topology[i+1];
    m_weights.emplace_back(Eigen::MatrixXd::NullaryExpr(b, a,[&](){
      return distribution(prng);
    }));
  }
}

template NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, std::mt19937& prng, size_t memory);

void NeuralNetwork::input(std::vector<double> input)
{
  auto output = this->output();
  for(size_t i=0; i<m_memory; ++i)
    input.push_back(output[output.size() - m_memory + i]);

  m_layers.front().input(convert(input));
}

double NeuralNetwork::output(size_t i) const
{
  // TODO: cache this
  return m_output(i);
}

std::vector<double> NeuralNetwork::output() const
{
  return convert(m_output);
}

void NeuralNetwork::feedForward()
{
  for(size_t i=0; i<m_weights.size(); ++i)
    m_layers[i+1].input(m_weights[i] * m_layers[i].output());

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

