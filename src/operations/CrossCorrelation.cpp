#include <libkann/operations/CrossCorrelation.hpp>

#include <libkann/operations/Impl.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  CrossCorrelationOperation::CrossCorrelationOperation(Vec2 kernel_size, Vec2 padding_size)
    : m_kernel_size(kernel_size), m_padding_size(padding_size) {}

  std::vector<Tensor> CrossCorrelationOperation::process(std::vector<Tensor> inputs) const
  {
    return operation_process_impl<2, 1>(std::move(inputs), [this](const Tensor& inputs, const Tensor& kernels) {
      return std::make_tuple(math::cross_correlate2d(inputs, kernels,
            inputs.shape().rank() - 3,
            kernels.shape().rank() - 3,
            1, false, false, m_padding_size));
    });
  }

  class CrossCorrelationGradientOperation : public Operation
  {
  public:
    CrossCorrelationGradientOperation(Vec2 kernel_size, Vec2 padding_size)
      : m_kernel_size(kernel_size), m_padding_size(padding_size) {}

  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override
    {
      return operation_process_impl<3, 2>(std::move(inputs), [this](const Tensor& inputs, const Tensor& kernels, const Tensor& output_gradients) {
        return std::make_tuple(
          math::convolve2d(output_gradients, kernels,       inputs.shape().rank() - 3, 1, kernels.shape().rank() - 3, false, true, m_kernel_size - m_padding_size - Vec2(1,1) ),
          math::cross_correlate2d(inputs, output_gradients, 1, kernels.shape().rank() - 3, inputs.shape().rank() - 3, false, true, m_padding_size)
        );
      });
    }

  private:
    Vec2 m_kernel_size;
    Vec2 m_padding_size;
  };

  operation_t CrossCorrelationOperation::differentiate() const
  {
    return std::make_shared<CrossCorrelationGradientOperation>(m_kernel_size, m_padding_size);
  }
}

