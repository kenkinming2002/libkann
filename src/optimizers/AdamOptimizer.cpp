#include <libkann/optimizers/AdamOptimizer.hpp>

#include <libkann/Math.hpp>

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
        MutableTensor::constant(shape, 0.0).as_const(),
        MutableTensor::constant(shape, 0.0).as_const(),
      };
    }

    // First and second moment respectively
    const Tensor& m = variable.optimizer_states[0];
    const Tensor& v = variable.optimizer_states[1];

    assert(variable.gradient);
    Tensor m_new = math::cwise(m, *variable.gradient, [this](float m, float gradient) { return m_beta1 * m + (1.0-m_beta1) * gradient * gradient; });
    Tensor v_new = math::cwise(v, *variable.gradient, [this](float v, float gradient) { return m_beta2 * v + (1.0-m_beta2) * gradient; });

    Tensor m_hat = math::scale(m_new, 1.0 / (1.0 - std::pow(m_beta1, (float)m_timestep)));
    Tensor v_hat = math::scale(v_new, 1.0 / (1.0 - std::pow(m_beta2, (float)m_timestep)));

    Tensor correction = math::scale(m_hat, m_alpha / ( math::norm(v_hat.as_ref()) + m_epsilon ));
    variable.value = math::sub(variable.value, correction);
    variable.optimizer_states = { std::move(m_new), std::move(v_new) };
  }
}

