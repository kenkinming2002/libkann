#include <libkann/LossFunction.hpp>

namespace kann
{
  tensor::Tensor<const float> LossFunction::forward(tensor::Tensor<const float> inputs)
  {
    const tensor::Shape inputs_shape = inputs.shape();

    const size_t batch_size = inputs.shape().dimension(0);
    const size_t input_size = inputs.shape().drop_front(1).size();

    assert(this->expected_outputs);
    auto _inputs           = inputs           .reshape(tensor::Shape(batch_size, input_size));
    auto _expected_outputs = expected_outputs->reshape(tensor::Shape(batch_size, input_size));
    auto _outputs          = this->loss_forward(std::move(_inputs), std::move(_expected_outputs));

    this->saved_tensors.clear();
    this->saved_tensors.reserve(1);
    this->saved_tensors.push_back(std::move(inputs));

    return _outputs.reshape(tensor::Shape(batch_size));
  }

  tensor::Tensor<const float> LossFunction::backward(tensor::Tensor<const float> output_gradients)
  {
    const tensor::Tensor<const float>& inputs       = this->saved_tensors[0];
    const tensor::Shape                inputs_shape = inputs.shape();

    const size_t batch_size = inputs.shape().dimension(0);
    const size_t input_size = inputs.shape().drop_front(1).size();

    assert(this->expected_outputs);
    auto _inputs           = inputs           .reshape(tensor::Shape(batch_size, input_size));
    auto _expected_outputs = expected_outputs->reshape(tensor::Shape(batch_size, input_size));
    auto _output_gradients = output_gradients .reshape(tensor::Shape(batch_size));
    auto _outputs          = this->loss_backward(std::move(_inputs), std::move(_expected_outputs), std::move(_output_gradients));

    return _outputs.reshape(inputs_shape);
  }
}
