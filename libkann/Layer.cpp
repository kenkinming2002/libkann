#include "Layer.hpp"

#include <libkann/ActivationFunction.hpp>

void Layer::input(Eigen::VectorXd input)
{
  m_input = std::move(input);
}

Eigen::VectorXd Layer::output() const
{
  // TODO: allow switching activation function
  return m_input.unaryExpr(&activation_function::tanh);
}

