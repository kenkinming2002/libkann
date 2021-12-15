#include <libkann/layers/WeightLayer.hpp>

#include <iostream>

namespace kann
{
  WeightLayer::WeightLayer(size_t inputSize, size_t outputSize)
    : Layer(inputSize * outputSize), m_inputSize(inputSize), m_outputSize(outputSize) {}

  std::unique_ptr<Layer> WeightLayer::clone() const
  {
    return std::make_unique<WeightLayer>(*this);
  }

  size_t WeightLayer::inputSize() const
  {
    return m_inputSize;
  }

  size_t WeightLayer::outputSize() const
  {
    return m_outputSize;
  }

  void WeightLayer::feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const
  {
    output = weight() * input;
  }

  void WeightLayer::backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const
  {
    layerGradient.resize(params().size());

    inputGradient = outputGradient * weight();
    weightGradient(layerGradient) = (input * outputGradient).transpose();
  }

}
