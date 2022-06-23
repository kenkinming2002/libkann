#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  SimpleOptimizer::SimpleOptimizer(float learningRate)
    : m_learningRate(learningRate) {}

  void SimpleOptimizer::optimize(Variable& variable) const
  {
    assert(variable.gradient);
    math::transform<1>(variable.value.as_ref(), {variable.gradient->as_const_ref()}, math::FMA(-m_learningRate));
  }
}
