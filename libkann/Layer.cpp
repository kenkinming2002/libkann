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

std::ostream& operator<<(std::ostream& os, const Layer& layer)
{
  os << "Input:\n";
  os << layer.m_input << '\n';
  os << "Output:\n";
  os << layer.output();

  return os;
}
