#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libtensor/Math.hpp>

namespace kann
{
  SimpleOptimizer::SimpleOptimizer(float learningRate)
    : m_learningRate(learningRate) {}

  void SimpleOptimizer::optimize(Variable& variable) const
  {
    math::transform<1>(variable.value.flatten(), {variable.gradient.flatten()}, math::FMA(-m_learningRate));
  }
}
