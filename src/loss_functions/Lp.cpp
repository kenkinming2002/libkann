#include <libkann/loss_functions/Lp.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  LpLossFunction::LpLossFunction(unsigned p)
    : m_p(p) {}

  static inline float pow_abs(float value, unsigned n)
  {
    value = std::abs(value);

    float result = value;
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

    MutableTensor outputs = MutableTensor::create(inputs.shape().front(1));
    outputs.fill(0.0);
    math::reduce<2>(outputs.as_ref(), {inputs.as_ref(), this->expected_outputs->as_ref()}, math::Direction::RIGHT, [this](float output, float input, float expected_output)
    {
      const float diff = input - expected_output;
      return output + pow_abs(diff, m_p);
    });
    return outputs.as_const();
  }

  Tensor LpLossFunction::backward(Tensor output_gradients)
  {
    assert(this->expected_outputs);
    const Tensor& inputs = saved_tensors[0];

    MutableTensor input_gradients = MutableTensor::create(inputs.shape());
    math::transform<2>(input_gradients.as_ref(), {inputs.as_ref(), this->expected_outputs->as_ref()},[this](float /*input_gradient*/, float input, const float expected_output)
    {
      const float diff = input - expected_output;
      return (m_p-1) * pow_abs(diff, m_p-1) * sgn(diff);
    });
    math::broadcast<1>(input_gradients.as_ref(), {output_gradients.as_ref()}, math::Direction::RIGHT, math::MUL);
    return input_gradients.as_const();
  }
}
