#include <libkann/optimizers/AdamOptimizer.hpp>

#include <libkann/Math.hpp>

#include <fmt/core.h>

namespace kann
{
  AdamOptimizer::AdamOptimizer(float alpha, float beta1, float beta2, float epsilon)
    : m_alpha(alpha), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon) {}

  void AdamOptimizer::step()
  {
    ++m_timestep;
  }

  void AdamOptimizer::optimize(Variable& variable) const
  {
    const Shape& shape = variable.value.shape();
    if(variable.optimizer_states.empty())
    {
      variable.optimizer_states = {
        MutableTensor::constant(shape, 0.0),
        MutableTensor::constant(shape, 0.0),
      };
      fmt::print("Create state\n");
    }

    // First and second moment respectively
    MutableTensor& m = variable.optimizer_states[0];
    MutableTensor& v = variable.optimizer_states[1];

    assert(variable.gradient);

    math::transform<1>(m.as_ref(), {variable.gradient->as_ref()}, [this](float m, float gradient) { return m_beta1 * m + (1.0-m_beta1) * gradient * gradient; });
    math::transform<1>(v.as_ref(), {variable.gradient->as_ref()}, [this](float v, float gradient) { return m_beta2 * v + (1.0-m_beta2) * gradient; });

    MutableTensor m_hat = MutableTensor::create(m.shape());
    MutableTensor v_hat = MutableTensor::create(v.shape());

    math::transform<1>(m_hat.as_ref(), {m.as_ref().as_const()}, math::SCALE(1.0 / (1.0 - std::pow(m_beta1, m_timestep))));
    math::transform<1>(v_hat.as_ref(), {v.as_ref().as_const()}, math::SCALE(1.0 / (1.0 - std::pow(m_beta2, m_timestep))));

    const float factor = -m_alpha / ( math::norm(v_hat.as_ref().as_const()) + m_epsilon );
    math::transform<1>(variable.value.as_ref(), {m_hat.as_ref().as_const()}, math::FMA(factor));
  }
}

