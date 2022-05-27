#include <libkann/operations/IdentityOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  IdentityOperation::IdentityOperation(size_t inputSize, size_t outputSize, size_t offset)
    : m_inputSize(inputSize), m_outputSize(outputSize), m_offset(offset) {}

  Tensor IdentityOperation::processImpl(const Tensor& input) const
  {
    Tensor output(m_outputSize);
    output.asArray().setZero();

    if(m_inputSize>=m_outputSize)
      output.asArray() = input.asArray().segment(m_offset, m_outputSize);
    else
      output.asArray().segment(m_offset, m_inputSize) = input.asArray();

    return output;
  }

  variable_t IdentityOperation::gradientsImpl(variable_t gradient, variable_t) const
  {
    auto output = std::make_shared<const Variable>(std::vector{std::move(gradient)}, std::make_shared<IdentityOperation>(m_outputSize, m_inputSize, m_offset));
    return output;
  }
}
