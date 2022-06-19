#include <libkann/operations/Subtract.hpp>

#include <libkann/operations/CWise.hpp>

namespace kann
{
  SubtractOperation::SubtractOperation(Shape shape)
    : m_shape(shape) {}

  std::vector<Tensor> SubtractOperation::process(std::vector<Tensor> inputs) const
  {
    return operation_process_cwise_impl<2,1>(std::move(inputs), m_shape, [](float a, float b) {
      return std::make_tuple(a-b);
    });
  }

  class SubtractGradientOperation : public Operation
  {
  public:
    SubtractGradientOperation(Shape shape)
      : m_shape(shape) {}

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const
    {
      return operation_process_cwise_impl<3,2>(std::move(inputs), m_shape, [](float a, float b, float output_gradient) {
        return std::make_tuple(output_gradient, -output_gradient);
      });
    }

  private:
    Shape m_shape;
  };

  operation_t SubtractOperation::differentiate() const
  {
    return std::make_shared<SubtractGradientOperation>(m_shape);
  }
}

