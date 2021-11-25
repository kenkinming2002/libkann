#include <libkann/Layer.hpp>

#include <libkann/ActivationFunction.hpp>

void Layer::feedForward(const ActivationFunction& activationFunction)
{
  m_output = m_input.unaryExpr([&](double val) { return activationFunction.normal(val); });
}

Eigen::VectorXd Layer::backPropagate(const Eigen::VectorXd& outputGradient, const ActivationFunction& activationFunction) const
{
  return m_input.unaryExpr([&](double val) { return activationFunction.derivative(val); }).cwiseProduct(outputGradient);
}

