#include <libkann/layers/ActivationLayer.hpp>

#include <functional>

namespace kann
{
  using namespace std::placeholders;

  ActivationLayer::ActivationLayer(size_t size, ActivationFunction activationFunction)
    : Layer(0), m_size(size), m_activationFunction(activationFunction) {}

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

  void ActivationLayer::feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const
  {
    output = input.unaryExpr(std::bind(&ActivationFunction::normal, &m_activationFunction, _1));
  }

  void ActivationLayer::backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const
  {
    inputGradient = input.unaryExpr(std::bind(&ActivationFunction::derivative, &m_activationFunction, _1)).transpose().cwiseProduct(outputGradient);
    layerGradient.resize(0); // We do not have any gradient
  }
}
