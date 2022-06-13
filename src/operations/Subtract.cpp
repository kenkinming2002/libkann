#include <libkann/operations/Subtract.hpp>

#include <libkann/operations/CWise.hpp>

namespace kann
{
  SubtractOperation::SubtractOperation(size_t size)
    : m_size(size) {}

  std::vector<tensor_t> SubtractOperation::process(std::vector<tensor_t> inputs) const
  {
    return operation_process_cwise_impl<2,1>(std::move(inputs), m_size, [](double a, double b) {
      return std::make_tuple(a-b);
    });
  }

  class SubtractGradientOperation : public Operation
  {
  public:
    SubtractGradientOperation(size_t size)
      : m_size(size) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const
    {
      return operation_process_cwise_impl<3,2>(std::move(inputs), m_size, [](double a, double b, double output_gradient) {
        return std::make_tuple(output_gradient, -output_gradient);
      });
    }

  private:
    size_t m_size;
  };

  operation_t SubtractOperation::differentiate() const
  {
    return std::make_shared<SubtractGradientOperation>(m_size);
  }
}

