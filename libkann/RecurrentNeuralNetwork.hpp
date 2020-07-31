#pragma once

#include <libkann/NeuralNetwork.hpp>

class RecurrentNeuralNetwork : public NeuralNetwork
{
public:
  LIBKANN_SYMEXPORT RecurrentNeuralNetwork(NeuralNetwork&& neuralNetwork, size_t memory);

public:
  LIBKANN_SYMEXPORT RecurrentNeuralNetwork(dynarray<size_t> topology, size_t memory);
  template<typename PRNG>
  LIBKANN_SYMEXPORT RecurrentNeuralNetwork(dynarray<size_t> topology, size_t memory, PRNG& prng);

public:
  using NeuralNetwork::input;
  using NeuralNetwork::output;

public:
  LIBKANN_SYMEXPORT void feedForward();

public:
  template<typename PRNG>
  LIBKANN_SYMEXPORT static RecurrentNeuralNetwork cross(const RecurrentNeuralNetwork& lhs, const RecurrentNeuralNetwork& rhs, 
      PRNG& prng, double mutationRate);

private:
  size_t m_memory;
};
