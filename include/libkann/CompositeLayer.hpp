#pragma once

#include <libkann/Layer.hpp>

#include <Eigen/Eigen>

template<typename Layer1, typename Layer2>
class CompositeLayer
{
public:
  CompositeLayer() = default;

public:
  CompositeLayer(Layer1 layer1, Layer2 layer2)
    : m_layer1(std::move(layer1)),
      m_layer2(std::move(layer2)) {}

public:
  Eigen::VectorXd feedForward(Eigen::MatrixXd input)
  {
    auto intermediate = m_layer1.feedForward(input);
    return m_layer2.feedForward(intermediate);
  }

  Eigen::RowVectorXd backPropagate(const Eigen::RowVectorXd& outputGradient)
  {
    auto intermediate = m_layer2.backPropagate(outputGradient);
    return m_layer1.backPropagate(intermediate);
  }

public:
  Layer1& layer1()             { return m_layer1; }
  const Layer1& layer1() const { return m_layer1; }

  Layer2& layer2()             { return m_layer2; }
  const Layer2& layer2() const { return m_layer2; }

public:
  size_t inputSize()  const { return m_layer1.inputSize(); }
  size_t outputSize() const { return m_layer2.outputSize(); }

private:
  Layer1 m_layer1;
  Layer2 m_layer2;
};

