#include <libkann/neural_networks/RecurrentNeuralNetwork.hpp>

#include <iostream>

namespace kann
{
  RecurrentNeuralNetwork::RecurrentNeuralNetwork(NeuralNetwork&& nn, size_t memory)
    : NeuralNetwork(std::move(nn)), m_memory(memory) {}

  RecurrentNeuralNetwork::RecurrentNeuralNetwork(size_t memory)
    : NeuralNetwork(), m_memory(memory) {}

  void RecurrentNeuralNetwork::feedForward(Eigen::VectorXd input)
  {
    Eigen::VectorXd realInput(this->inputSize());
    realInput << input, this->output().tail(m_memory);
    NeuralNetwork::feedForward(realInput);
  }

  RecurrentNeuralNetwork RecurrentNeuralNetwork::cross(const RecurrentNeuralNetwork& other, std::default_random_engine& prng, double mutationRate) const
  {
    assert(m_memory == other.m_memory);
    auto result = NeuralNetwork::cross(other, prng, mutationRate);
    return RecurrentNeuralNetwork(std::move(result), other.m_memory);
  }
}
