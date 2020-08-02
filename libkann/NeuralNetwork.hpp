#pragma once

#include <libkann/export.hpp>

#include <libkann/utilities/dynarray.hpp>
#include <libkann/Layer.hpp>
#include <libkann/Connection.hpp>

#include <Eigen/Eigen>

#include <ostream>
#include <vector>
#include <random>
#include <type_traits>
#include <initializer_list>

class NeuralNetwork
{
public:
  LIBKANN_SYMEXPORT NeuralNetwork(dynarray<size_t> topology);
  template<typename PRNG>
  LIBKANN_SYMEXPORT NeuralNetwork(dynarray<size_t> topology, PRNG& prng);

public:
  template<typename T>
  LIBKANN_SYMEXPORT void input(T i, double v) { return this->input(static_cast<size_t>(i), v); }
  LIBKANN_SYMEXPORT void input(size_t i, double v) { m_layers.front().input()(i) = v; }

public:
  template<typename T>
  LIBKANN_SYMEXPORT double output(T i) const { return this->output(static_cast<size_t>(i)); }
  LIBKANN_SYMEXPORT double output(size_t i) const { return m_output(i); }

public:
  LIBKANN_SYMEXPORT const auto& topology() const { return m_topology; }

public:
  LIBKANN_SYMEXPORT void feedForward();
  LIBKANN_SYMEXPORT void backPropagate(const Eigen::VectorXd& expectedOutput);
  LIBKANN_SYMEXPORT void train(double learningRate);

public:
  template<typename PRNG>
  LIBKANN_SYMEXPORT static NeuralNetwork cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, PRNG& prng, double mutationRate);

public:
  dynarray<size_t> m_topology;
  dynarray<Layer> m_layers;
  dynarray<Connection> m_connections;

  Eigen::VectorXd m_output;
};
