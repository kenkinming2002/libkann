#include <libkann/optimizers/AdamOptimizer.hpp>

#include <libtensor/Map.hpp>
#include <libtensor/Norm.hpp>

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
    const tensor::Shape& shape = variable.shape;
    if(variable.optimizer_states.empty())
    {
      tensor::Tensor<float> m = tensor::Tensor<float>::create(shape);
      tensor::Tensor<float> v = tensor::Tensor<float>::create(shape);
      m.fill(0.0f);
      v.fill(0.0f);

      // Hack to work around the fact that initializer list return by const and
      // thus does not work for move only types
      variable.optimizer_states.clear();
      variable.optimizer_states.reserve(2);
      variable.optimizer_states.push_back(std::move(m));
      variable.optimizer_states.push_back(std::move(v));
    }

    auto& m = variable.optimizer_states[0];
    auto& v = variable.optimizer_states[1];

    m = tensor::binary_map<float>(m, variable.gradient, [this](float m, float gradient) { return m_beta1 * m + (1.0-m_beta1) * gradient; });
    v = tensor::binary_map<float>(m, variable.gradient, [this](float v, float gradient) { return m_beta2 * v + (1.0-m_beta2) * gradient * gradient; });

    auto m_hat = tensor::unary_map<float>(m, [factor = 1.0 / (1.0 - std::pow(m_beta1, m_timestep))](float m) { return m * factor; });
    auto v_hat = tensor::unary_map<float>(v, [factor = 1.0 / (1.0 - std::pow(m_beta2, m_timestep))](float v) { return v * factor; });

    const float factor = m_alpha / ( tensor::norm<float>(v_hat) + m_epsilon );
    variable.value = tensor::binary_map<float>(variable.value, m_hat, [factor](float value, float m_hat) {
      return value - factor * m_hat;
    });
  }
}

