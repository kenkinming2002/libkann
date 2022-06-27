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
    const size_t batch_size = inputs.shape().dimension(0);
    const Shape  shape      = inputs.shape().drop_front(1);
    const size_t size       = shape.size();

    Tensor<float> outputs = Tensor<float>::create(Shape{batch_size});

    assert(this->expected_outputs);
    auto _inputs            = inputs           .reshape(Shape{batch_size, size});
    auto _expected_outputs  = expected_outputs->reshape(Shape{batch_size, size});
    auto _outputs           = outputs          .reshape(Shape{batch_size});

    outputs.fill(0.0);
    math::reduce<2>(_outputs, {_inputs, _expected_outputs}, math::Direction::RIGHT, [this](float output, float input, float expected_output)
    {
      const float diff = input - expected_output;
      return output + pow_abs(diff, m_p);
    });

    saved_tensors.clear();
    saved_tensors.reserve(1);
    saved_tensors.push_back(std::move(inputs));
    return outputs;
  }

  Tensor<float> LpLossFunction::backward(Tensor<float> output_gradients)
  {
    Tensor<float> inputs = std::move(saved_tensors[0]);

    const size_t batch_size = inputs.shape().dimension(0);
    const Shape  shape      = inputs.shape().drop_front(1);
    const size_t size       = shape.size();

    Tensor<float> input_gradients = Tensor<float>::create(inputs.shape());

    assert(this->expected_outputs);
    auto _inputs            = inputs           .reshape(Shape{batch_size, size});
    auto _input_gradients   = input_gradients  .reshape(Shape{batch_size, size});
    auto _expected_outputs  = expected_outputs->reshape(Shape{batch_size, size});
    auto _output_gradients  = output_gradients .reshape(Shape{batch_size});

    math::transform<2>(_input_gradients.flatten(), {_inputs.flatten(), _expected_outputs.flatten()}, [this](float /*input_gradient*/, float input, const float expected_output)
    {
      const float diff = input - expected_output;
      return m_p * pow_abs(diff, m_p-1) * sgn(diff);
    });
    math::broadcast<1>(_input_gradients, { _output_gradients }, math::Direction::RIGHT, math::MUL);
    return input_gradients;
  }
}
