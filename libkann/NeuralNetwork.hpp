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
  using random_engine_type = std::mt19937;
  using seed_type = typename random_engine_type::result_type;

public:
  LIBKANN_SYMEXPORT NeuralNetwork(const std::vector<size_t>& topology);
  LIBKANN_SYMEXPORT NeuralNetwork(const std::vector<size_t>& topology, seed_type seed);

public:
  // TODO: Consider optimizing this
  LIBKANN_SYMEXPORT void input(const std::vector<double>& input);
  LIBKANN_SYMEXPORT std::vector<double> output() const;

public:
  LIBKANN_SYMEXPORT void feedForward();
  LIBKANN_SYMEXPORT void backPropagate(const std::vector<double>& input);

public:
  LIBKANN_SYMEXPORT static NeuralNetwork cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, seed_type seed, double mutationRate);

public:
  LIBKANN_SYMEXPORT friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& neuralNetwork);

private:
  std::vector<size_t> m_topology;
  std::vector<Layer> m_layers;
  std::vector<Eigen::MatrixXd> m_weights;
};
