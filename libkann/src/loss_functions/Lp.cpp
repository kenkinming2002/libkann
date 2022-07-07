#include <libkann/loss_functions/Lp.hpp>

#include <libtensor/Reduce.hpp>
#include <libtensor/Broadcast.hpp>
#include <libtensor/Map.hpp>

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

  tensor::Tensor<const float> LpLossFunction::forward(tensor::Tensor<const float> inputs)
  {
    this->saved_tensors.clear();
    this->saved_tensors.reserve(1);
    this->saved_tensors.push_back(inputs);

    assert(this->expected_outputs);
    auto expected_outputs = *this->expected_outputs;

    inputs           = inputs          .flatten(tensor::flatten_single, this->shape);
    expected_outputs = expected_outputs.flatten(tensor::flatten_single, this->shape);

    auto tmps    = tensor::binary_map(inputs, expected_outputs, [this](float input, float expected_output) {
      const float diff = input - expected_output;
      return pow_abs(diff, m_p);
    });
    auto outputs = tensor::reduce<tensor::Direction::RIGHT, float>(tmps);
    return outputs;
  }

  tensor::Tensor<const float> LpLossFunction::backward(tensor::Tensor<const float> output_gradients)
  {
    assert(this->expected_outputs);
    auto inputs           = this->saved_tensors[0];
    auto expected_outputs = *this->expected_outputs;

    inputs           = inputs          .flatten(tensor::flatten_single, this->shape);
    expected_outputs = expected_outputs.flatten(tensor::flatten_single, this->shape);

    auto tmps            = tensor::binary_map(inputs, expected_outputs, [this](float input, float expected_output) {
      const float diff = input - expected_output;
      return m_p * pow_abs(diff, m_p-1) * sgn(diff);
    });
    auto input_gradients = tensor::broadcast_mul<tensor::Direction::RIGHT, float>(tmps, output_gradients);
    return input_gradients;
  }
}
