#include <libkann/loss_functions/CrossEntropy.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  Tensor<float> CrossEntropyLossFunction::forward(Tensor<float> inputs)
  {
    assert(this->expected_outputs);

    Tensor<float> outputs = Tensor<float>::create(inputs.as_ref().shape().front(1));
    outputs.as_ref().fill(0.0);
    math::reduce<2>(outputs.as_ref(), {inputs.as_const_ref(), this->expected_outputs->as_const_ref()}, math::Direction::RIGHT, [](float output, float input, float expected_output)
    {
      // Assume inputs is properly normalized via the use of a normalization layer such as softmax
      // Entropy/Information of input * Probability of getting such input
      return output - std::log(input) * expected_output;
    });

    saved_tensors.clear();
    saved_tensors.reserve(1);
    saved_tensors.push_back(std::move(inputs));
    return outputs;
  }

  Tensor<float> CrossEntropyLossFunction::backward(Tensor<float> output_gradients)
  {
    assert(this->expected_outputs);
    const Tensor<float>& inputs = saved_tensors[0];

    Tensor<float> input_gradients = Tensor<float>::create(inputs.as_ref().shape());
    math::transform<2>(input_gradients.as_ref(), {inputs.as_const_ref(), this->expected_outputs->as_const_ref()},[](float /*input_gradient*/, float input, float expected_output)
    {
      return - expected_output / input;
    });
    math::broadcast<1>(input_gradients.as_ref(), {output_gradients.as_const_ref()}, math::Direction::RIGHT, math::MUL);
    return input_gradients;
  }
}

