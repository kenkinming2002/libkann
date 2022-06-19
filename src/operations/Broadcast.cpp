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
    return operation_process_impl<1, 1>(std::move(inputs), [this](const Tensor& input) {
      return std::make_tuple(math::broadcast(input, m_shape));
    });
  }

  class BroadcastGradientOperation : public Operation
  {
  public:
    BroadcastGradientOperation(Shape shape) : m_shape(shape) {}

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override
    {
      return operation_process_impl<2, 1>(std::move(inputs), [this](const Tensor& input, const Tensor& output_gradient) {
        return std::make_tuple(math::reduce(output_gradient, m_shape));
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
