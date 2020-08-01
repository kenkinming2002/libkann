#include "Layer.hpp"

#include <libkann/ActivationFunction.hpp>

Eigen::VectorXd Layer::output() const
{
  // TODO: allow switching activation function
  return m_input.unaryExpr(&activation::function::tanh);
}

Eigen::VectorXd Layer::backPropagate(Eigen::VectorXd outputGradient) const
{
  return m_input.unaryExpr(&activation::derivative::tanh).cwiseProduct(outputGradient);
}

