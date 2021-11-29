#include <libkann/ActivationLayer.hpp>

#include <functional>

using namespace std::placeholders;

ActivationLayer::ActivationLayer(size_t size)
{
  m_input = Eigen::VectorXd(size);
}

Eigen::VectorXd ActivationLayer::feedForward(Eigen::VectorXd input)
{
  m_input = input;
  return m_input.unaryExpr(std::bind(&ActivationFunction::normal, &m_activationFunction, _1));
}

Eigen::RowVectorXd ActivationLayer::backPropagate(const Eigen::RowVectorXd& outputGradient)
{
  return m_input.unaryExpr(std::bind(&ActivationFunction::derivative, &m_activationFunction, _1)).transpose().cwiseProduct(outputGradient);
}

size_t ActivationLayer::inputSize() const
{
  return m_input.size();
}

size_t ActivationLayer::outputSize() const
{
  return m_input.size();
}
