#include "Connection.hpp"

#include <random>
#include <iostream>

Connection::Connection(size_t prevLayerSize, size_t nextLayerSize)
  : m_weight(nextLayerSize, prevLayerSize), m_weightGradient(Eigen::MatrixXd::Zero(nextLayerSize, prevLayerSize)) {}

template<typename PRNG>
void Connection::randomize(PRNG& prng)
{
  std::uniform_real_distribution<double> weightDistribution(-1.0, 1.0);
  m_weight = Eigen::MatrixXd::NullaryExpr(m_weight.rows(), m_weight.cols(),[&](){ return weightDistribution(prng); });
}
template void Connection::randomize(std::mt19937& prng);

void Connection::feedForward(const Layer& prevLayer, Layer& nextLayer)
{
  nextLayer.input() = m_weight * prevLayer.output();
}

Eigen::VectorXd Connection::backPropagate(const Eigen::VectorXd& input, const Eigen::VectorXd& outputGradient)
{
  Eigen::MatrixXd weightGradient(m_weightGradient.rows(), m_weightGradient.cols());
  for(size_t y=0; y<weightGradient.rows(); ++y)
    for(size_t x=0; x<weightGradient.cols(); ++x)
      weightGradient(y, x) = input(x) * outputGradient(y);

  m_weightGradient += weightGradient;

  Eigen::VectorXd inputGradient = m_weight.transpose() * outputGradient;
  return inputGradient;
}

void Connection::train(double learningRate)
{
  m_weight -= learningRate * m_weightGradient;
  m_weightGradient = Eigen::MatrixXd::Zero(m_weightGradient.rows(), m_weightGradient.cols());
}

template<typename PRNG>
Connection Connection::cross(const Connection& lhs, const Connection& rhs, PRNG& prng, double mutationRate)
{
  assert(lhs.m_weight.size() != 0);
  assert(lhs.m_weight.size() == rhs.m_weight.size());

  std::uniform_int_distribution<> distribution(0, 1);
  std::uniform_real_distribution<double> mutationDistribution(0.0, 1.0);
  std::uniform_real_distribution<double> weightDistribution(-1.0, 1.0);

  Connection result = lhs;;
  for(long i=0; i<lhs.m_weight.size(); ++i)
  {
    result.m_weight.data()[i] = distribution(prng) == 0 ? lhs.m_weight.data()[i] : rhs.m_weight.data()[i];
    if(mutationDistribution(prng) < mutationRate)
      result.m_weight.data()[i] = weightDistribution(prng);
  }

  return result;
}
template Connection Connection::cross(const Connection& lhs, const Connection& rhs, std::mt19937& prng, double mutationRate);
