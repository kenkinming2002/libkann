#pragma once

#include <libkann/export.hpp>
#include <libkann/ActivationFunction.hpp>

#include <Eigen/Eigen>

class Layer
{
public:
  Layer() = default;
  Layer(size_t size) : m_input(size), m_output(size) {}

public:
  Eigen::VectorXd& input() { return m_input; }
  const Eigen::VectorXd& input() const { return m_input; }
  const Eigen::VectorXd& output() const { return m_output; }

public:
  void feedForward(const ActivationFunction& activationFunction);
  Eigen::VectorXd backPropagate(const Eigen::VectorXd& outputGradient, const ActivationFunction& activationFunction) const;


private:
  Eigen::VectorXd m_input;
  Eigen::VectorXd m_output;
};
