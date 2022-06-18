#include <libkann/operations/Broadcast.hpp>

#include <libkann/operations/Impl.hpp>

#include <libkann/Utils.hpp>
#include <libkann/Math.hpp>

namespace kann
{
  BroadcastOperation::BroadcastOperation(Shape shape)
    : m_shape(shape) {}

  std::vector<Tensor> BroadcastOperation::process(std::vector<Tensor> inputs) const
  {
    return operation_process_impl<1, 1>(std::move(inputs), [this](const Tensor& input)
    {
      MutableTensor output = MutableTensor::create(Shape::concat(m_shape, input.shape()));
      {
        const size_t size = m_shape.size();
        MutableTensorRef _output = output.as_ref().reshape(Shape::concat(Shape(size), input.shape()));
        for(size_t i=0; i<size; ++i)
          utils::copy(input.as_ref(), _output[i]);
      }
      return std::make_tuple(std::move(output).as_const());
    });
  }

  class BroadcastGradientOperation : public Operation
  {
  public:
    BroadcastGradientOperation(Shape shape) : m_shape(shape) {}

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override
    {
      return operation_process_impl<2, 1>(std::move(inputs), [this](const Tensor& input, const Tensor& output_gradient)
      {
        MutableTensor input_gradient = MutableTensor::create(output_gradient.shape().drop_front(m_shape.rank()));
        input_gradient.fill(0.0);
        {
          const size_t size = m_shape.size();
          TensorRef _output_gradients = output_gradient.as_ref().reshape(Shape::concat(Shape(size), input.shape()));
          for(size_t i=0; i<size; ++i)
            math::add_to(input_gradient.as_ref(), _output_gradients[i]);
        }
        return std::make_tuple(std::move(input_gradient).as_const());
      });
    }

  private:
    Shape m_shape;
  };

  operation_t BroadcastOperation::differentiate() const
  {
    return std::make_shared<BroadcastGradientOperation>(m_shape);
  }
}
