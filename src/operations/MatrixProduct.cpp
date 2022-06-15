#include <libkann/operations/MatrixProduct.hpp>

#include <libkann/operations/Impl.hpp>

namespace kann
{
  std::vector<Tensor> MatrixProductOperation::process(std::vector<Tensor> inputs) const
  {
    return operation_process_impl<2, 1>(std::move(inputs), [](const Tensor& a, const Tensor& b) {
      return std::make_tuple(math::product(a, b,
        a.shape().dimension_count() - 1,
        b.shape().dimension_count() - 1,
        1, false, false));
    });
  }

  class MatrixProductGradientOperation : public Operation
  {
  public:
    std::vector<Tensor> process(std::vector<Tensor> inputs) const override
    {
      return operation_process_impl<3, 2>(std::move(inputs), [](const Tensor& a, const Tensor& b, const Tensor& output_gradient) {
        return std::make_tuple(
          math::product(output_gradient, b, a.shape().dimension_count() - 1, 1, b.shape().dimension_count() - 1, false, true),
          math::product(a, output_gradient, 1, b.shape().dimension_count() - 1, a.shape().dimension_count() - 1, true, false));
      });
    }
  };

  operation_t MatrixProductOperation::differentiate() const
  {
    return std::make_shared<MatrixProductGradientOperation>();
  }
}

