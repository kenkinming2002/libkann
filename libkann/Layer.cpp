#include "Layer.hpp"

#include <libkann/ActivationFunction.hpp>

void Layer::feedForward()
{
  m_output = m_input.unaryExpr(&activation::function::tanh);
}

Eigen::VectorXd Layer::backPropagate(const Eigen::VectorXd& outputGradient) const
{
  return m_input.unaryExpr(&activation::derivative::tanh).cwiseProduct(outputGradient);
}

