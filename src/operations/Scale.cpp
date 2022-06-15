#include <libkann/operations/Scale.hpp>

#include <libkann/operations/CWise.hpp>

namespace kann
{
  ScaleOperation::ScaleOperation(Shape shape, double val)
    : m_shape(shape), m_val(val) {}

  std::vector<Tensor> ScaleOperation::process(std::vector<Tensor> inputs) const
  {
    return operation_process_cwise_impl<1,1>(std::move(inputs), m_shape, [this](double input) {
      return std::make_tuple(input * m_val);
    });
  }

  class ScaleGradientOperation : public Operation
  {
  public:
    ScaleGradientOperation(Shape shape, double val)
      : m_shape(shape), m_val(val) {}

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const
    {
      return operation_process_cwise_impl<2,1>(std::move(inputs), m_shape, [this](double input, double output_gradient) {
        return std::make_tuple(output_gradient * m_val);
      });
    }

  private:
    Shape m_shape;
    double m_val;
  };

  operation_t ScaleOperation::differentiate() const
  {
    return std::make_shared<ScaleGradientOperation>(m_shape, m_val);
  }
}
