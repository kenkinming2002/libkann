#pragma once

#include <libkann/export.hpp>

#include <libkann/Layer.hpp>

#include <Eigen/Eigen>

#include <ostream>
#include <vector>
#include <random>

class NeuralNetwork
{
public:
  LIBKANN_SYMEXPORT NeuralNetwork(const std::vector<size_t>& topology);

  template<typename PRNG>
  LIBKANN_SYMEXPORT NeuralNetwork(const std::vector<size_t>& topology, PRNG& prng);

public:
  // TODO: Consider optimizing this
  LIBKANN_SYMEXPORT void input(const std::vector<double>& input);
  LIBKANN_SYMEXPORT std::vector<double> output() const;

public:
  LIBKANN_SYMEXPORT void feedForward();
  LIBKANN_SYMEXPORT void backPropagate(const std::vector<double>& input);

public:
  template<typename PRNG>
  LIBKANN_SYMEXPORT static NeuralNetwork cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, PRNG& prng, double mutationRate);

public:
  LIBKANN_SYMEXPORT friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& neuralNetwork);

private:
  std::vector<size_t> m_topology;
  std::vector<Layer> m_layers;
  std::vector<Eigen::MatrixXd> m_weights;
};
