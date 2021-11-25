#pragma once

#include <libkann/export.hpp>
#include <libkann/Layer.hpp>

#include <Eigen/Eigen>

#include <random>
#include <assert.h>

class WeightLayer
{
public:
  WeightLayer() = default;

public:
  LIBKANN_SYMEXPORT WeightLayer(size_t inputSize, size_t outputSize);

public:
  LIBKANN_SYMEXPORT Eigen::VectorXd feedForward(Eigen::MatrixXd input);
  LIBKANN_SYMEXPORT Eigen::RowVectorXd backPropagate(const Eigen::RowVectorXd& outputGradient);

public:
  LIBKANN_SYMEXPORT size_t inputSize() const;
  LIBKANN_SYMEXPORT size_t outputSize() const;

public:
  LIBKANN_SYMEXPORT void train(double learningRate);

public:
  template<typename PRNG>
  void randomize(PRNG& prng);

public:
  template<typename PRNG>
  static WeightLayer cross(const WeightLayer& lhs, const WeightLayer& rhs, PRNG& prng, double mutationRate);

public:
  void weight(Eigen::MatrixXd weight)
  {
    assert(m_weight.rows() == weight.rows());
    assert(m_weight.cols() == weight.cols());
    m_weight = weight;
  }

  Eigen::MatrixXd weight() const
  {
    return m_weight;
  }

private:
  Eigen::VectorXd m_input;
  Eigen::MatrixXd m_weight;
  Eigen::MatrixXd m_weightGradient;
};

static_assert(isLayer<WeightLayer>);
