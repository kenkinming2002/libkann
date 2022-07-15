#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libtensor/Map.hpp>

namespace kann
{
  SimpleOptimizer::SimpleOptimizer(float learningRate)
    : m_learningRate(learningRate) {}

  void SimpleOptimizer::optimize(Variable& variable) const
  {
    variable.value = tensor::binary_map<float>(variable.value, variable.gradient, [this](float value, float gradient) {
      return value - m_learningRate * gradient;
    });
  }
}
