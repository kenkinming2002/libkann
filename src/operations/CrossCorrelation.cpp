#include <libkann/operations/CrossCorrelation.hpp>

#include <libkann/operations/Impl.hpp>

namespace kann
{
  CrossCorrelationOperation::CrossCorrelationOperation(Vec2 input_size, Vec2 output_size, Vec2 kernel_size)
    : m_input_size(input_size), m_output_size(output_size), m_kernel_size(kernel_size) {}

  std::vector<tensor_t> CrossCorrelationOperation::process(std::vector<tensor_t> inputs) const
  {
    return operation_process_impl<2, 1>(std::move(inputs), [this](const Tensor& input, const Tensor& kernel) {
      return std::make_tuple(Tensor::cross_correlate(input, kernel, m_input_size, m_output_size, m_kernel_size));
    });
  }

  class CrossCorrelationGradientOperation : public Operation
  {
  public:
    CrossCorrelationGradientOperation(Vec2 input_size, Vec2 output_size, Vec2 kernel_size)
      : m_input_size(input_size), m_output_size(output_size), m_kernel_size(kernel_size) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      return operation_process_impl<3, 2>(std::move(inputs), [this](const Tensor& input, const Tensor& kernel, const Tensor& output_gradient) {
        return std::make_tuple(
          Tensor::convolve(output_gradient, kernel, m_output_size, m_input_size, m_kernel_size),
          Tensor::cross_correlate(input, output_gradient, m_input_size, m_kernel_size, m_output_size)
        );
      });
    }

  private:
    Vec2 m_input_size, m_output_size, m_kernel_size;
  };

  operation_t CrossCorrelationOperation::differentiate() const
  {
    return std::make_shared<CrossCorrelationGradientOperation>(m_input_size, m_output_size, m_kernel_size);
  }
}

