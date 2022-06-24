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
    const Shape& shape = variable.value.as_ref().shape();
    if(variable.optimizer_states.empty())
    {
      Tensor<float> m = Tensor<float>::create(shape);
      Tensor<float> v = Tensor<float>::create(shape);
      m.as_ref().fill(0.0f);
      v.as_ref().fill(0.0f);

      // Hack to work around the fact that initializer list return by const and
      // thus does not work for move only types
      variable.optimizer_states.clear();
      variable.optimizer_states.reserve(2);
      variable.optimizer_states.push_back(std::move(m));
      variable.optimizer_states.push_back(std::move(v));
    }

    // First and second moment respectively
    Tensor<float>& m = variable.optimizer_states[0];
    Tensor<float>& v = variable.optimizer_states[1];

    assert(variable.gradient);

    math::transform<1>(m.as_ref(), {variable.gradient->as_const_ref()}, [this](float m, float gradient) { return m_beta1 * m + (1.0-m_beta1) * gradient; });
    math::transform<1>(v.as_ref(), {variable.gradient->as_const_ref()}, [this](float v, float gradient) { return m_beta2 * v + (1.0-m_beta2) * gradient * gradient; });

    Tensor<float> m_hat = Tensor<float>::create(m.as_ref().shape());
    Tensor<float> v_hat = Tensor<float>::create(v.as_ref().shape());

    math::transform<1>(m_hat.as_ref(), {m.as_const_ref()}, math::SCALE(1.0 / (1.0 - std::pow(m_beta1, m_timestep))));
    math::transform<1>(v_hat.as_ref(), {v.as_const_ref()}, math::SCALE(1.0 / (1.0 - std::pow(m_beta2, m_timestep))));

    const float factor = -m_alpha / ( math::norm(v_hat.as_const_ref()) + m_epsilon );
    math::transform<1>(variable.value.as_ref(), {m_hat.as_const_ref()}, math::FMA(factor));
  }
}

