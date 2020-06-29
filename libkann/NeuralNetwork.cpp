#include "NeuralNetwork.hpp"

#include <libkann/ActivationFunction.hpp>

#include <cassert>

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

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology) : m_topology(topology)
{
  // Layers
  for(size_t size: topology)
    m_layers.emplace_back(size);

  for(size_t i=0; i<topology.size()-1; i++)
  {
    size_t a = topology[i], b = topology[i+1];
    m_weights.emplace_back(Eigen::MatrixXd(b, a));
  }
}

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, seed_type seed) : m_topology(topology)
{
  // Layers
  for(size_t size: topology)
    m_layers.emplace_back(size);

  // Weights
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  for(size_t i=0; i<topology.size()-1; i++)
  {
    size_t a = topology[i], b = topology[i+1];
    m_weights.emplace_back(Eigen::MatrixXd::NullaryExpr(b, a,[&](){
      return distribution(generator);
    }));
  }
}

void NeuralNetwork::input(const std::vector<double>& input)
{
  m_layers.front().input(convert(input));
}

std::vector<double> NeuralNetwork::output() const
{
  return convert(m_layers.back().output());
}

void NeuralNetwork::feedForward()
{
  for(size_t i=0; i<m_weights.size(); ++i)
    m_layers[i+1].input(m_weights[i] * m_layers[i].output());
}

NeuralNetwork NeuralNetwork::cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, seed_type seed)
{
  std::mt19937 generator(seed);
  std::uniform_int_distribution<> distribution(0, 1);

  std::uniform_real_distribution<double> mutationDistribution(0.0, 1.0);
  std::uniform_real_distribution<double> weightDistribution(-1.0, 1.0);

  static constexpr double MUTATION_PROBABILITY = 0.01;

  NeuralNetwork output(lhs.m_topology);
  for(size_t i=0; i<lhs.m_weights.size(); ++i)
  {
    auto& lhsWeight = lhs.m_weights[i];
    auto& rhsWeight = rhs.m_weights[i]; 
    auto& outputWeight = output.m_weights[i];

    for(size_t j=0; j<lhsWeight.size(); ++j)
    {
      if(distribution(generator) == 0)
        outputWeight.data()[j] = lhsWeight.data()[j];
      else
        outputWeight.data()[j] = rhsWeight.data()[j];

      if(mutationDistribution(generator) < MUTATION_PROBABILITY)
        outputWeight.data()[j] = weightDistribution(generator);
    }
  }

  return output;
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& neuralNetwork)
{
  for(size_t i=0; i<neuralNetwork.m_weights.size(); ++i)
  {
    os << neuralNetwork.m_layers[i] << '\n';
    os << "==========\n";
    os << neuralNetwork.m_weights[i] << '\n';
    os << "==========\n";
  }

  os << neuralNetwork.m_layers.back();

  return os;
}
