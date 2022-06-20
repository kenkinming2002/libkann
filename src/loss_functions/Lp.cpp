#include <libkann/loss_functions/Lp.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  LpLossFunction::LpLossFunction(unsigned p)
    : m_p(p) {}

  static inline float pow_abs(float value, unsigned n)
  {
    float result = std::abs(value);
    for(unsigned i=1; i<n; ++i)
      result *= value;
    return result;
  }

  static inline float sgn(float value)
  {
    return value >= 0.0f ? 1.0f : -1.0f;
  }

  Tensor LpLossFunction::forward(Tensor inputs)
  {
    assert(this->expected_outputs);
    saved_tensors = { inputs };
    return math::cwise(inputs, *this->expected_outputs, [this](float input, float expected_output) {
      const float diff = input - expected_output;
      return pow_abs(diff, m_p);
    });
  }

  Tensor LpLossFunction::backward(Tensor output_gradients)
  {
    assert(this->expected_outputs);
    const Tensor& inputs = saved_tensors[0];
    return math::cwise(inputs, *this->expected_outputs, output_gradients, [this](float input, float expected_output, float output_gradient) {
      const float diff = input - expected_output;
      return (m_p-1) * pow_abs(diff, m_p-1) * sgn(diff) * output_gradient;
    });
  }
}
