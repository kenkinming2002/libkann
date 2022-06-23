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

  Tensor<float> LpLossFunction::forward(Tensor<float> inputs)
  {
    assert(this->expected_outputs);

    Tensor<float> outputs = Tensor<float>::create(inputs.as_ref().shape().front(1));
    outputs.as_ref().fill(0.0);
    math::reduce<2>(outputs.as_ref(), {inputs.as_const_ref(), this->expected_outputs->as_const_ref()}, math::Direction::RIGHT, [this](float output, float input, float expected_output)
    {
      const float diff = input - expected_output;
      return output + pow_abs(diff, m_p);
    });

    saved_tensors = { std::move(inputs) };
    return outputs;
  }

  Tensor<float> LpLossFunction::backward(Tensor<float> output_gradients)
  {
    assert(this->expected_outputs);
    Tensor<float> inputs = std::move(saved_tensors[0]);

    Tensor<float> input_gradients = Tensor<float>::create(inputs.as_ref().shape());
    math::transform<2>(input_gradients.as_ref(), {inputs.as_const_ref(), this->expected_outputs->as_const_ref()},[this](float /*input_gradient*/, float input, const float expected_output)
    {
      const float diff = input - expected_output;
      return (m_p-1) * pow_abs(diff, m_p-1) * sgn(diff);
    });
    math::broadcast<1>(input_gradients.as_ref(), {output_gradients.as_const_ref()}, math::Direction::RIGHT, math::MUL);
    return input_gradients;
  }
}
