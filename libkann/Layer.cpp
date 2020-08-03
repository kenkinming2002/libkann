#include "Layer.hpp"

#include <libkann/ActivationFunction.hpp>

void Layer::feedForward(const ActivationFunction& activationFunction)
{
  m_output = m_input.unaryExpr(activationFunction.normal);
}

Eigen::VectorXd Layer::backPropagate(const Eigen::VectorXd& outputGradient, const ActivationFunction& activationFunction) const
{
  return m_input.unaryExpr(activationFunction.derivative).cwiseProduct(outputGradient);
}

