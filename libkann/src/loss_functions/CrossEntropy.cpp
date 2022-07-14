#include <libkann/loss_functions/CrossEntropy.hpp>

#include <libtensor/Reduce.hpp>
#include <libtensor/Broadcast.hpp>
#include <libtensor/Map.hpp>

namespace kann
{
  tensor::Tensor<const float> CrossEntropyLossFunction::forward(tensor::Tensor<const float> inputs)
  {
    this->saved_tensors.clear();
    this->saved_tensors.reserve(1);
    this->saved_tensors.push_back(inputs);

    assert(this->expected_outputs);
    auto expected_outputs = *this->expected_outputs;

    inputs           = inputs          .flatten(tensor::Hint::single(), tensor::Hint::from_shape(this->shape));
    expected_outputs = expected_outputs.flatten(tensor::Hint::single(), tensor::Hint::from_shape(this->shape));

    auto tmps    = tensor::binary_map(inputs, expected_outputs, [](float input, float expected_output) { return -std::log(input) * expected_output; });
    auto outputs = tensor::reduce<tensor::Direction::RIGHT, float>(tmps);
    return outputs;
  }

  tensor::Tensor<const float> CrossEntropyLossFunction::backward(tensor::Tensor<const float> output_gradients)
  {
    assert(this->expected_outputs);
    auto inputs           = this->saved_tensors[0];
    auto expected_outputs = *this->expected_outputs;

    inputs           = inputs          .flatten(tensor::Hint::single(), tensor::Hint::from_shape(this->shape));
    expected_outputs = expected_outputs.flatten(tensor::Hint::single(), tensor::Hint::from_shape(this->shape));

    auto tmps            = tensor::binary_map(inputs, expected_outputs, [](float input, float expected_output) { return -expected_output / input; });
    auto input_gradients = tensor::broadcast_mul<tensor::Direction::RIGHT, float>(tmps, output_gradients);
    return input_gradients;
  }
}

