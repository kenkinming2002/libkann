#include <libkann/operations/Split2.hpp>

#include <libkann/operations/Impl.hpp>

#include <tuple>

namespace kann
{
  Split2Operation::Split2Operation(size_t size1, size_t size2)
    : m_size1(size1), m_size2(size2) {}

  std::vector<tensor_t> Split2Operation::process(std::vector<tensor_t> inputs) const
  {
    return operation_process_impl<1,2>(std::move(inputs), [this](const Tensor& input)
    {
      assert(input.size() == m_size1 + m_size2);
      Tensor output1(m_size1), output2(m_size2);
      output1.asArray() = input.asArray().segment(0,       m_size1);
      output2.asArray() = input.asArray().segment(m_size1, m_size2);
      return std::make_tuple(std::move(output1), std::move(output2));
    });
  }

  class Split2GradientOperation : public Operation
  {
  public:
    Split2GradientOperation(size_t size1, size_t size2)
      : m_size1(size1), m_size2(size2) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      return operation_process_impl<3,1>(std::move(inputs), [this](const Tensor& input, const Tensor& output_gradient1, const Tensor& output_gradient2)
      {
        assert(output_gradient1.size() == m_size1);
        assert(output_gradient2.size() == m_size2);

        Tensor input_gradient(m_size1 + m_size2);
        input_gradient.asArray().segment(0,       m_size1) = output_gradient1.asArray();
        input_gradient.asArray().segment(m_size1, m_size2) = output_gradient2.asArray();
        return std::make_tuple(std::move(input_gradient));
      });
    }

  private:
    size_t m_size1, m_size2;
  };

  operation_t Split2Operation::differentiate() const
  {
    return std::make_shared<Split2GradientOperation>(m_size1, m_size2);
  }
}
