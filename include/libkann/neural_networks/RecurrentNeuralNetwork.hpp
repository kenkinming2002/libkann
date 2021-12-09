#pragma once

#include <libkann/neural_networks/NeuralNetwork.hpp>

namespace kann
{
  class RecurrentNeuralNetwork : public NeuralNetwork
  {
  public:
    LIBKANN_SYMEXPORT RecurrentNeuralNetwork(NeuralNetwork&& nn, size_t memory);
    LIBKANN_SYMEXPORT RecurrentNeuralNetwork(size_t memory);

  public:
    LIBKANN_SYMEXPORT void feedForward(Eigen::VectorXd input);

  public:
    LIBKANN_SYMEXPORT RecurrentNeuralNetwork cross(const RecurrentNeuralNetwork& other, std::default_random_engine& prng, double mutationRate) const;

  private:
    size_t m_memory;
  };
}
