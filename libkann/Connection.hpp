#pragma once

#include <libkann/Layer.hpp>

#include <Eigen/Eigen>

#include <cstddef>

class Connection
{
public:
  Connection() = default;
  Connection(size_t prevLayerSize, size_t nextLayerSize);

public:
  template<typename PRNG>
  void randomize(PRNG& prng);

public:
  void feedForward(const Layer& prevLayer, Layer& nextLayer);
  Eigen::VectorXd backPropagate(const Eigen::VectorXd& input, const Eigen::VectorXd& outputGradient);
  void train(double learningRate);

public:
  template<typename PRNG>
  static Connection cross(const Connection& lhs, const Connection& rhs, PRNG& prng, double mutationRate);

private:
  Eigen::MatrixXd m_weight;
  Eigen::MatrixXd m_weightGradient;
};
