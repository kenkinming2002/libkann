#include <libkann/operations/Scale.hpp>

#include <libkann/operations/CWise.hpp>

namespace kann
{
  ScaleOperation::ScaleOperation(size_t size, double val)
    : m_size(size), m_val(val) {}

  std::vector<tensor_t> ScaleOperation::process(std::vector<tensor_t> inputs) const
  {
    return operation_process_cwise_impl<1,1>(std::move(inputs), m_size, [this](double input) {
      return std::make_tuple(input * m_val);
    });
  }

  class ScaleGradientOperation : public Operation
  {
  public:
    ScaleGradientOperation(size_t size, double val)
      : m_size(size), m_val(val) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const
    {
      return operation_process_cwise_impl<2,1>(std::move(inputs), m_size, [this](double input, double output_gradient) {
        return std::make_tuple(output_gradient * m_val);
      });
    }

  private:
    size_t m_size;
    double m_val;
  };

  operation_t ScaleOperation::differentiate() const
  {
    return std::make_shared<ScaleGradientOperation>(m_size, m_val);
  }
}
