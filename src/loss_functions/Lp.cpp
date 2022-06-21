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

    MutableTensor tmps = MutableTensor::create(inputs.shape());
    math::transform2(inputs.as_ref(), this->expected_outputs->as_ref(), tmps.as_ref(), math::BinaryOperation::create([this](const float& input, const float& expected_output, float& tmp)
    {
      const float diff = input - expected_output;
      tmp = pow_abs(diff, m_p);
    }));

    MutableTensor outputs = MutableTensor::create(inputs.shape().front(1));
    outputs.fill(0.0);
    math::reduce(tmps.as_const().as_ref(), outputs.as_ref(), math::Direction::RIGHT, math::ADD);
    return outputs.as_const();
  }

  Tensor LpLossFunction::backward(Tensor output_gradients)
  {
    assert(this->expected_outputs);
    const Tensor& inputs = saved_tensors[0];

    MutableTensor input_gradients = MutableTensor::create(inputs.shape());
    math::transform2(inputs.as_ref(), this->expected_outputs->as_ref(), input_gradients.as_ref(), math::BinaryOperation::create([this](const float& input, const float& expected_output, float& input_gradient)
    {
      const float diff = input - expected_output;
      input_gradient = (m_p-1) * pow_abs(diff, m_p-1) * sgn(diff);
    }));
    math::broadcast(output_gradients.as_ref(), input_gradients.as_ref(), math::Direction::RIGHT, math::MUL);
    return input_gradients.as_const();
  }
}
