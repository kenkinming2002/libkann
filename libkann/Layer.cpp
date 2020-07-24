#include "Layer.hpp"

#include <libkann/ActivationFunction.hpp>

Eigen::VectorXd Layer::output() const
{
  // TODO: allow switching activation function
  return m_input.unaryExpr(&activation_function::tanh);
}

