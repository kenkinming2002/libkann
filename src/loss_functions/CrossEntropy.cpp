#include <libkann/loss_functions/CrossEntropy.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  Tensor CrossEntropyLossFunction::forward(Tensor inputs)
  {
    assert(this->expected_outputs);
    saved_tensors = { inputs };

    MutableTensor outputs = MutableTensor::create(inputs.shape().front(1));
    outputs.fill(0.0);
    math::reduce<2>(outputs.as_ref(), {inputs.as_ref(), this->expected_outputs->as_ref()}, math::Direction::RIGHT, [](float output, float input, float expected_output)
    {
      // Assume inputs is properly normalized via the use of a normalization layer such as softmax
      // Entropy/Information of input * Probability of getting such input
      return output - std::log(input) * expected_output;
    });
    return outputs.as_const();
  }

  Tensor CrossEntropyLossFunction::backward(Tensor output_gradients)
  {
    assert(this->expected_outputs);
    const Tensor& inputs = saved_tensors[0];

    MutableTensor input_gradients = MutableTensor::create(inputs.shape());
    math::transform<2>(input_gradients.as_ref(), {inputs.as_ref(), this->expected_outputs->as_ref()},[](float /*input_gradient*/, float input, float expected_output)
    {
      return - expected_output / input;
    });
    math::broadcast<1>(input_gradients.as_ref(), {output_gradients.as_ref()}, math::Direction::RIGHT, math::MUL);
    return input_gradients.as_const();
  }
}

