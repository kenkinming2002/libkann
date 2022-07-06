#include <libkann/loss_functions/CrossEntropy.hpp>

#include <libtensor/Math.hpp>

namespace kann
{
  tensor::Tensor<const float> CrossEntropyLossFunction::loss_forward(tensor::Tensor<const float> inputs, tensor::Tensor<const float> expected_outputs)
  {
    const size_t batch_size = inputs.shape().dimension(0);
    tensor::Tensor<float> outputs = tensor::Tensor<float>::create(tensor::Shape{batch_size});

    outputs.fill(0.0);
    tensor::math::reduce<2>(outputs, {inputs, expected_outputs}, tensor::math::Direction::RIGHT, [](float output, float input, float expected_output) {
      // Assume inputs is properly normalized via the use of a normalization layer such as softmax
      // Entropy/Information of input * Probability of getting such input
      return output - std::log(input) * expected_output;
    });

    return outputs;
  }

  tensor::Tensor<const float> CrossEntropyLossFunction::loss_backward(tensor::Tensor<const float> inputs, tensor::Tensor<const float> expected_outputs, tensor::Tensor<const float> output_gradients)
  {
    tensor::Tensor<float> input_gradients = tensor::Tensor<float>::create(inputs.shape());

    tensor::math::transform<2>(input_gradients.flatten(), {inputs.flatten(), expected_outputs.flatten()}, [](float /*input_gradient*/, float input, const float expected_output)
    {
      return - expected_output / input;
    });
    tensor::math::broadcast<1>(input_gradients, { output_gradients }, tensor::math::Direction::RIGHT, tensor::math::MUL);
    return input_gradients;
  }
}

