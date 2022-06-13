#include <libkann/operations/Concat2.hpp>

#include <libkann/operations/Impl.hpp>

#include <tuple>

namespace kann
{
  Concat2Operation::Concat2Operation(size_t size1, size_t size2)
    : m_size1(size1), m_size2(size2) {}

  std::vector<tensor_t> Concat2Operation::process(std::vector<tensor_t> inputs) const
  {
    return operation_process_impl<2,1>(std::move(inputs), [this](const Tensor& input1, const Tensor& input2)
    {
      assert(input1.size() == m_size1);
      assert(input2.size() == m_size2);

      Tensor output(m_size1 + m_size2);
      output.asArray().segment(0,       m_size1) = input1.asArray();
      output.asArray().segment(m_size1, m_size2) = input2.asArray();
      return std::make_tuple(std::move(output));
    });
  }

  class Concat2GradientOperation : public Operation
  {
  public:
    Concat2GradientOperation(size_t size1, size_t size2)
      : m_size1(size1), m_size2(size2) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      return operation_process_impl<3,2>(std::move(inputs), [this](const Tensor& input1, const Tensor& input2, const Tensor& output_gradient)
      {
        assert(output_gradient.size() == m_size1 + m_size2);
        Tensor input_gradient1(m_size1), input_gradient2(m_size2);
        input_gradient1.asArray() = output_gradient.asArray().segment(0,       m_size1);
        input_gradient2.asArray() = output_gradient.asArray().segment(m_size1, m_size2);
        return std::make_tuple(std::move(input_gradient1), std::move(input_gradient2));
      });
    }

  private:
    size_t m_size1, m_size2;
  };

  operation_t Concat2Operation::differentiate() const
  {
    return std::make_shared<Concat2GradientOperation>(m_size1, m_size2);
  }
}

