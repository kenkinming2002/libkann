#include "RecurrentNeuralNetwork.hpp"

#include <iostream>

namespace
{
  dynarray<size_t> makeTopology(dynarray<size_t> topology, size_t memory)
  {
    topology.front() += memory;
    topology.back()  += memory; 
    return topology;
  }
}

RecurrentNeuralNetwork::RecurrentNeuralNetwork(NeuralNetwork&& neuralNetwork, size_t memory)
  : NeuralNetwork(std::move(neuralNetwork)), m_memory(memory) {}

RecurrentNeuralNetwork::RecurrentNeuralNetwork(dynarray<size_t> topology, size_t memory)
  : NeuralNetwork(makeTopology(std::move(topology), memory)), m_memory(memory) {}

template<typename PRNG>
RecurrentNeuralNetwork::RecurrentNeuralNetwork(dynarray<size_t> topology, size_t memory, PRNG& prng)
  : NeuralNetwork(makeTopology(std::move(topology), memory), prng), m_memory(memory) {}

template RecurrentNeuralNetwork::RecurrentNeuralNetwork(dynarray<size_t> topology, size_t memory, std::mt19937& prng);

void RecurrentNeuralNetwork::feedForward()
{
  size_t inputSize  = this->topology().front();
  size_t outputSize = this->topology().back();
  for(size_t i=0; i<m_memory; ++i)
    this->input(inputSize - m_memory + i, this->output(outputSize - m_memory + i));

  NeuralNetwork::feedForward();
}

template<typename PRNG>
RecurrentNeuralNetwork RecurrentNeuralNetwork::cross(const RecurrentNeuralNetwork& lhs, const RecurrentNeuralNetwork& rhs, 
    PRNG& prng, double mutationRate)
{
  assert(lhs.m_memory == rhs.m_memory);

  auto result = NeuralNetwork::cross(lhs, rhs, prng, mutationRate);
  return RecurrentNeuralNetwork(std::move(result), lhs.m_memory);
}

template RecurrentNeuralNetwork RecurrentNeuralNetwork::cross<std::mt19937>(const RecurrentNeuralNetwork& lhs, const RecurrentNeuralNetwork& rhs, std::mt19937& prng, double mutationRate);
