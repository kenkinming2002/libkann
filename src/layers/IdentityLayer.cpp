#include <libkann/layers/IdentityLayer.hpp>

namespace kann
{
  using namespace std::placeholders;

  IdentityLayer::IdentityLayer(size_t size)
    : Layer(0), m_size(size) {}

  std::unique_ptr<Layer> IdentityLayer::clone() const
  {
    return std::make_unique<IdentityLayer>(*this);
  }

  size_t IdentityLayer::inputSize() const
  {
    return m_size;
  }

  size_t IdentityLayer::outputSize() const
  {
    return m_size;
  }

  void IdentityLayer::feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const
  {
    output = input;
  }

  void IdentityLayer::backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const
  {
    inputGradient = outputGradient;
    layerGradient.resize(0); // We do not have any gradient
  }
}

