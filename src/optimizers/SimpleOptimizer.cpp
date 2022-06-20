#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  SimpleOptimizer::SimpleOptimizer(float learningRate)
    : m_learningRate(learningRate) {}

  void SimpleOptimizer::optimize(Variable& variable) const
  {
    assert(variable.gradient);
    variable.value = math::cwise(variable.value, *variable.gradient, [this](double value, double gradient) {
      return value - gradient * m_learningRate;
    });
  }
}
