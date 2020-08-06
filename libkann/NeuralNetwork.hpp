#pragma once

#include <libkann/export.hpp>

#include <libkann/serialization/Eigen.hpp>
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
  NeuralNetwork() = default;

public:
  LIBKANN_SYMEXPORT NeuralNetwork(dynarray<size_t> topology, ActivationFunction activationFunction);
  LIBKANN_SYMEXPORT NeuralNetwork(dynarray<size_t> topology, dynarray<Eigen::MatrixXd> weights, ActivationFunction activationFunction);
  template<typename PRNG>
  LIBKANN_SYMEXPORT NeuralNetwork(dynarray<size_t> topology, PRNG& prng, ActivationFunction activationFunction);

public:
  LIBKANN_SYMEXPORT size_t inputSize() const { return m_layers.front().size(); }

  template<typename T>
  LIBKANN_SYMEXPORT void input(T i, double v) { return this->input(static_cast<size_t>(i), v); }
  LIBKANN_SYMEXPORT void input(size_t i, double v) { m_layers.front().input()(i) = v; }

public:
  LIBKANN_SYMEXPORT size_t outputSize() const { return m_layers.back().size(); }

  template<typename T>
  LIBKANN_SYMEXPORT double output(T i) const { return this->output(static_cast<size_t>(i)); }
  LIBKANN_SYMEXPORT double output(size_t i) const { return m_output(i); }

public:
  LIBKANN_SYMEXPORT void feedForward();
  LIBKANN_SYMEXPORT void backPropagate(const Eigen::VectorXd& expectedOutput);
  LIBKANN_SYMEXPORT void train(double learningRate);

public:
  template<typename PRNG>
  LIBKANN_SYMEXPORT static NeuralNetwork cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, PRNG& prng, double mutationRate);

private:
  dynarray<Layer> m_layers;
  dynarray<Connection> m_connections;

private:
  ActivationFunction m_activationFunction;

private:
  Eigen::VectorXd m_output;

public:
  template<typename Archive>
  void save(Archive& archive) const
  {
    // topology
    dynarray<size_t> topology(m_layers.size());
    std::transform(m_layers.begin(), m_layers.end(), topology.begin(), [](const auto& layer){ return layer.size(); });
    archive(topology);

    // weights
    dynarray<Eigen::MatrixXd> weights(m_connections.size());
    std::transform(m_connections.begin(), m_connections.end(), weights.begin(), [](const auto& connection){ return connection.weight(); });
    archive(weights);

    // activation function
    archive(m_activationFunction.type);

  }

  template<typename Archive>
  void load(Archive& archive)
  {
    // topology
    dynarray<size_t> topology;
    archive(topology);

    // weights
    dynarray<Eigen::MatrixXd> weights;
    archive(weights);

    // activation function
    ActivationFunction::Type type;
    archive(type);

    *this = NeuralNetwork(std::move(topology), std::move(weights), ActivationFunction(type));

  }
};
