#include <libkann/loss_functions/CrossEntropy.hpp>

#include <libtensor/Math.hpp>

namespace kann
{
  tensor::Tensor<const float> CrossEntropyLossFunction::forward(tensor::Tensor<const float> inputs)
  {
    const size_t batch_size = inputs.shape().dimension(0);
    const tensor::Shape  shape      = inputs.shape().drop_front(1);
    const size_t size       = shape.size();

    tensor::Tensor<float> outputs = tensor::Tensor<float>::create(tensor::Shape{batch_size});

    assert(this->expected_outputs);
    auto _inputs            = inputs           .reshape(tensor::Shape{batch_size, size});
    auto _expected_outputs  = expected_outputs->reshape(tensor::Shape{batch_size, size});
    auto _outputs           = outputs          .reshape(tensor::Shape{batch_size});

    outputs.fill(0.0);
    tensor::math::reduce<2>(_outputs, {_inputs, _expected_outputs}, tensor::math::Direction::RIGHT, [](float output, float input, float expected_output)
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

  tensor::Tensor<const float> CrossEntropyLossFunction::backward(tensor::Tensor<const float> output_gradients)
  {
    tensor::Tensor<const float> inputs = std::move(saved_tensors[0]);

    const size_t batch_size = inputs.shape().dimension(0);
    const tensor::Shape  shape      = inputs.shape().drop_front(1);
    const size_t size       = shape.size();

    tensor::Tensor<float> input_gradients = tensor::Tensor<float>::create(inputs.shape());

    assert(this->expected_outputs);
    auto _inputs            = inputs           .reshape(tensor::Shape{batch_size, size});
    auto _input_gradients   = input_gradients  .reshape(tensor::Shape{batch_size, size});
    auto _expected_outputs  = expected_outputs->reshape(tensor::Shape{batch_size, size});
    auto _output_gradients  = output_gradients .reshape(tensor::Shape{batch_size});

    tensor::math::transform<2>(_input_gradients.flatten(), {_inputs.flatten(), _expected_outputs.flatten()}, [](float /*input_gradient*/, float input, const float expected_output)
    {
      return - expected_output / input;
    });
    tensor::math::broadcast<1>(_input_gradients, { _output_gradients }, tensor::math::Direction::RIGHT, tensor::math::MUL);
    return input_gradients;
  }
}

