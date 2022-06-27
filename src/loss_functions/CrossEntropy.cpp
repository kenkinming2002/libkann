#include <libkann/loss_functions/CrossEntropy.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  Tensor<float> CrossEntropyLossFunction::forward(Tensor<float> inputs)
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
    math::reduce<2>(_outputs, {_inputs, _expected_outputs}, math::Direction::RIGHT, [](float output, float input, float expected_output)
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

    math::transform<2>(_input_gradients.flatten(), {_inputs.flatten(), _expected_outputs.flatten()}, [](float /*input_gradient*/, float input, const float expected_output)
    {
      return - expected_output / input;
    });
    math::broadcast<1>(_input_gradients, { _output_gradients }, math::Direction::RIGHT, math::MUL);
    return input_gradients;
  }
}

