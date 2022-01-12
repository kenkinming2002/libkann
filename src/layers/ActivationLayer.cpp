#include <libkann/layers/ActivationLayer.hpp>

#include <functional>

namespace kann
{
  using namespace std::placeholders;

  ActivationLayer::ActivationLayer(size_t size, ActivationFunction activationFunction)
    : m_size(size), m_activationFunction(activationFunction) {}

  std::unique_ptr<Layer> ActivationLayer::clone() const
  {
    return std::make_unique<ActivationLayer>(*this);
  }

  size_t ActivationLayer::inputSize() const
  {
    return m_size;
  }

  size_t ActivationLayer::outputSize() const
  {
    return m_size;
  }

  Eigen::VectorXd ActivationLayer::feedForward()
  {
    return input().unaryExpr(std::bind(&ActivationFunction::normal, &m_activationFunction, _1));
  }

  Eigen::VectorXd ActivationLayer::backPropagate()
  {
    return input().unaryExpr(std::bind(&ActivationFunction::derivative, &m_activationFunction, _1)).cwiseProduct(outputGradient());
  }

  std::vector<std::span<double>> ActivationLayer::params()
  {
    return {};
  }

  std::vector<std::span<const double>> ActivationLayer::params() const
  {
    return {};
  }

  std::vector<std::span<double>> ActivationLayer::paramsGradient()
  {
    return {};
  }

  std::vector<std::span<const double>> ActivationLayer::paramsGradient() const
  {
    return {};
  }
}
