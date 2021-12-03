#include <libkann/ActivationLayer.hpp>

#include <functional>

namespace kann
{
  using namespace std::placeholders;

  ActivationLayer::ActivationLayer(size_t size, ActivationFunction activationFunction)
  {
    m_input              = Eigen::VectorXd(size);
    m_activationFunction = activationFunction;
  }

  size_t ActivationLayer::inputSize() const
  {
    return m_input.size();
  }

  size_t ActivationLayer::outputSize() const
  {
    return m_input.size();
  }

  void ActivationLayer::randomize(std::default_random_engine& engine) {}

  Eigen::VectorXd ActivationLayer::feedForward(Eigen::VectorXd input)
  {
    m_input = input;
    return m_input.unaryExpr(std::bind(&ActivationFunction::normal, &m_activationFunction, _1));
  }

  Eigen::RowVectorXd ActivationLayer::backPropagate(const Eigen::RowVectorXd& outputGradient)
  {
    return m_input.unaryExpr(std::bind(&ActivationFunction::derivative, &m_activationFunction, _1)).transpose().cwiseProduct(outputGradient);
  }

  void ActivationLayer::train(double learningRate) {}

  std::unique_ptr<Layer> ActivationLayer::cross(const Layer& other, std::default_random_engine& engine, double mutationRate) const
  {
    return std::make_unique<ActivationLayer>(*this);
  }
}
