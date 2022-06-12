#include <libkann/operations/ConvolutionOperation.hpp>

namespace kann
{
  CrossCorrelationOperation::CrossCorrelationOperation(Vec2 input_size, Vec2 output_size, Vec2 kernel_size)
    : m_input_size(input_size), m_output_size(output_size), m_kernel_size(kernel_size) {}

  auto CrossCorrelationOperation::process_impl(inputs_t inputs) const -> outputs_t
  {
    const auto& [input, kernel] = inputs;
    Tensor output = Tensor::cross_correlate(*input, *kernel, m_input_size, m_output_size, m_kernel_size);
    return { std::move(output) };
  }

  class CrossCorrelationGradientOperation : public OperationImpl<CrossCorrelationGradientOperation, 3, 2>
  {
  public:
    CrossCorrelationGradientOperation(Vec2 input_size, Vec2 output_size, Vec2 kernel_size)
      : m_input_size(input_size), m_output_size(output_size), m_kernel_size(kernel_size) {}

  public:
    outputs_t process_impl(inputs_t inputs) const
    {
      const auto& [input, kernel, output_gradient] = inputs;
      Tensor input_gradient  = Tensor::convolve(*output_gradient, *kernel, m_output_size, m_input_size, m_kernel_size);
      Tensor kernel_gradient = Tensor::cross_correlate(*input, *output_gradient, m_input_size, m_kernel_size, m_output_size);
      return { std::move(input_gradient), std::move(kernel_gradient) };
    }

  private:
    Vec2 m_input_size, m_output_size, m_kernel_size;
  };

  operation_t CrossCorrelationOperation::differentiate() const
  {
    return std::make_shared<CrossCorrelationGradientOperation>(m_input_size, m_output_size, m_kernel_size);
  }
}

