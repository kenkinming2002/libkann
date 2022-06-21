#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  SimpleOptimizer::SimpleOptimizer(float learningRate)
    : m_learningRate(learningRate) {}

  void SimpleOptimizer::optimize(Variable& variable) const
  {
    assert(variable.gradient);
    math::transform(variable.gradient->as_ref(), variable.value.as_ref(), math::FMA(-m_learningRate));
  }
}
