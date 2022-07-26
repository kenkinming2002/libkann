#include <libkann/loss_functions/CrossEntropy.hpp>

#include <libtensor/Reduce.hpp>
#include <libtensor/Broadcast.hpp>
#include <libtensor/Map.hpp>

namespace kann
{
  tensor::Tensor<float> CrossEntropyLossFunction::forward(tensor::Tensor<float> inputs)
  {
    this->saved_tensors.clear();
    this->saved_tensors.reserve(1);
    this->saved_tensors.push_back(inputs);

    assert(this->expected_outputs);
    auto expected_outputs = *this->expected_outputs;
    auto tmps    = tensor::binary_map(inputs, expected_outputs, [](float input, float expected_output) {
      return -std::log(input) * expected_output;
    });
    auto outputs = tensor::reduce_inner<float>(tmps, -1);
    return outputs;
  }

  tensor::Tensor<float> CrossEntropyLossFunction::backward(tensor::Tensor<float> output_gradients)
  {
    assert(this->expected_outputs);
    auto inputs           = this->saved_tensors[0];
    auto expected_outputs = *this->expected_outputs;

    inputs           = inputs          .flatten(tensor::Hint::single(), tensor::Hint::from_shape(this->shape));
    expected_outputs = expected_outputs.flatten(tensor::Hint::single(), tensor::Hint::from_shape(this->shape));

    auto tmps            = tensor::binary_map(inputs, expected_outputs, [](float input, float expected_output) { return -expected_output / input; });
    auto input_gradients = tensor::broadcast_mul_inner<float>(tmps, output_gradients);

    input_gradients = input_gradients.unflatten(tensor::Hint::single(), tensor::Hint::from_shape(this->shape));
    return input_gradients;
  }
}

