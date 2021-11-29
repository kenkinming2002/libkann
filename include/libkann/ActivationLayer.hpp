#pragma once

#include <libkann/export.hpp>
#include <libkann/Layer.hpp>
#include <libkann/ActivationFunction.hpp>

#include <Eigen/Eigen>

class ActivationLayer
{
public:
  ActivationLayer() = default;

public:
  LIBKANN_SYMEXPORT ActivationLayer(size_t size);

public:
  LIBKANN_SYMEXPORT Eigen::VectorXd feedForward(Eigen::VectorXd input);
  LIBKANN_SYMEXPORT Eigen::RowVectorXd backPropagate(const Eigen::RowVectorXd& outputGradient);

public:
  LIBKANN_SYMEXPORT size_t inputSize() const;
  LIBKANN_SYMEXPORT size_t outputSize() const;

public:
  void activationFunction(ActivationFunction activationFunction)
  {
    m_activationFunction = activationFunction;
  }

  ActivationFunction activationFunction() const
  {
    return m_activationFunction;
  }

private:
  ActivationFunction m_activationFunction;
  Eigen::VectorXd m_input;
};

static_assert(isLayer<ActivationLayer>);
