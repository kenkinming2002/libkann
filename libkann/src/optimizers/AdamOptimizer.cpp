#include <libkann/optimizers/AdamOptimizer.hpp>

#include <libtensor/Math.hpp>

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
    const Shape& shape = variable.shape;
    if(variable.optimizer_states.empty())
    {
      Tensor<float> m = Tensor<float>::create(shape);
      Tensor<float> v = Tensor<float>::create(shape);
      m.fill(0.0f);
      v.fill(0.0f);

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

    math::transform<1>(m.flatten(), {variable.gradient.flatten()}, [this](float m, float gradient) { return m_beta1 * m + (1.0-m_beta1) * gradient; });
    math::transform<1>(v.flatten(), {variable.gradient.flatten()}, [this](float v, float gradient) { return m_beta2 * v + (1.0-m_beta2) * gradient * gradient; });

    Tensor<float> m_hat = Tensor<float>::create(m.shape());
    Tensor<float> v_hat = Tensor<float>::create(v.shape());

    math::transform<1>(m_hat.flatten(), {m.flatten()}, math::SCALE(1.0 / (1.0 - std::pow(m_beta1, m_timestep))));
    math::transform<1>(v_hat.flatten(), {v.flatten()}, math::SCALE(1.0 / (1.0 - std::pow(m_beta2, m_timestep))));

    const float factor = -m_alpha / ( math::norm(v_hat.flatten()) + m_epsilon );
    math::transform<1>(variable.value.flatten(), {m_hat.flatten()}, math::FMA(factor));
  }
}

