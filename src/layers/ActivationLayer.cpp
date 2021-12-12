#include <libkann/layers/ActivationLayer.hpp>

#include <functional>

namespace kann
{
  using namespace std::placeholders;

  ActivationLayer::ActivationLayer(size_t size, ActivationFunction activationFunction)
    : m_size(size), m_activationFunction(activationFunction) {}

  size_t ActivationLayer::inputSize() const
  {
    return m_size;
  }

  size_t ActivationLayer::outputSize() const
  {
    return m_size;
  }

  void ActivationLayer::randomize(std::default_random_engine& engine) {}

  Eigen::VectorXd ActivationLayer::feedForward(const Eigen::VectorXd& input)
  {
    return input.unaryExpr(std::bind(&ActivationFunction::normal, &m_activationFunction, _1));
  }

  Eigen::RowVectorXd ActivationLayer::backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient)
  {
    return input.unaryExpr(std::bind(&ActivationFunction::derivative, &m_activationFunction, _1)).transpose().cwiseProduct(outputGradient);
  }

  void ActivationLayer::train(double learningRate) {}

  std::unique_ptr<Layer> ActivationLayer::cross(const Layer& other, std::default_random_engine& engine, double mutationRate) const
  {
    return std::make_unique<ActivationLayer>(*this);
  }
}
