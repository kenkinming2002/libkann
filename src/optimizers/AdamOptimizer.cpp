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
    const MutableTensor& m = variable.optimizer_states[0];
    const MutableTensor& v = variable.optimizer_states[1];

    assert(variable.gradient);
    Tensor m_new = math::cwise(m.as_const(), *variable.gradient, [this](float m, float gradient) { return m_beta1 * m + (1.0-m_beta1) * gradient * gradient; });
    Tensor v_new = math::cwise(v.as_const(), *variable.gradient, [this](float v, float gradient) { return m_beta2 * v + (1.0-m_beta2) * gradient; });

    Tensor m_hat = math::scale(m_new, 1.0 / (1.0 - std::pow(m_beta1, (float)m_timestep)));
    Tensor v_hat = math::scale(v_new, 1.0 / (1.0 - std::pow(m_beta2, (float)m_timestep)));

    const float factor = -m_alpha / ( math::norm(v_hat.as_ref()) + m_epsilon );
    math::transform(m_hat.as_ref(), variable.value.as_ref(), math::FMA(factor));

    math::transform(m_new.as_ref(), m.as_ref(), math::STORE);
    math::transform(v_new.as_ref(), v.as_ref(), math::STORE);
  }
}

