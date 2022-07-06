#include <libkann/loss_functions/Lp.hpp>

#include <libtensor/Math.hpp>

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

  tensor::Tensor<const float> LpLossFunction::loss_forward(tensor::Tensor<const float> inputs, tensor::Tensor<const float> expected_outputs)
  {
    const size_t batch_size = inputs.dimension(0);
    tensor::Tensor<float> outputs = tensor::Tensor<float>::create(tensor::Shape(batch_size));

    outputs.fill(0.0f);
    tensor::math::reduce<2>(outputs, {inputs, expected_outputs}, tensor::math::Direction::RIGHT, [this](float output, float input, float expected_output) {
      const float diff = input - expected_output;
      return output + pow_abs(diff, m_p);
    });

    return outputs;
  }

  tensor::Tensor<const float> LpLossFunction::loss_backward(tensor::Tensor<const float> inputs, tensor::Tensor<const float> expected_outputs, tensor::Tensor<const float> output_gradients)
  {
    tensor::Tensor<float> input_gradients = tensor::Tensor<float>::create(inputs.shape());

    tensor::math::transform<2>(input_gradients.flatten(), {inputs.flatten(), expected_outputs.flatten()}, [this](float /*input_gradient*/, float input, const float expected_output) {
      const float diff = input - expected_output;
      return m_p * pow_abs(diff, m_p-1) * sgn(diff);
    });
    tensor::math::broadcast<1>(input_gradients, {output_gradients}, tensor::math::Direction::RIGHT, tensor::math::MUL);

    return input_gradients;
  }
}
