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
    const size_t batch_size = inputs.shape().dimension(0);

    Tensor tmp = math::cwise(inputs, *this->expected_outputs, [this](float input, float expected_output) {
      const float diff = input - expected_output;
      return pow_abs(diff, m_p);
    });

    MutableTensor result = MutableTensor::create(Shape(batch_size));
    result.fill(0.0);

    for(size_t i=0; i<batch_size; ++i)
      for(size_t j=0; j<tmp[i].size(); ++j)
        result.get(i) += tmp[i].get(j);

    return result.as_const();
  }

  Tensor LpLossFunction::backward(Tensor output_gradients)
  {
    assert(this->expected_outputs);
    const Tensor& inputs = saved_tensors[0];
    const size_t batch_size = inputs.shape().dimension(0);

    Tensor tmp = math::cwise(inputs, *this->expected_outputs, [this](float input, float expected_output) {
      const float diff = input - expected_output;
      return (m_p-1) * pow_abs(diff, m_p-1) * sgn(diff);
    });

    MutableTensor result = MutableTensor::create(tmp.shape());
    for(size_t i=0; i<batch_size; ++i)
      for(size_t j=0; j<tmp[i].size(); ++j)
        result[i].get(j) = tmp[i].get(j) * output_gradients.get(i);

    return result.as_const();
  }
}
