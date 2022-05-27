#include <libkann/operations/ConvolutionOperation.hpp>

#include <libkann/operations/DeconvolutionOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  ConvolutionOperation::ConvolutionOperation(size_t inputWidth, size_t inputHeight, size_t kernelSize)
    : m_inputWidth(inputWidth), m_inputHeight(inputHeight), m_kernelSize(kernelSize)
  {
    assert(m_inputWidth  > m_kernelSize);
    assert(m_inputHeight > m_kernelSize);
  }

  Tensor ConvolutionOperation::processImpl(const Tensor& input, const Tensor& kernel) const
  {
    const size_t outputWidth  = m_inputWidth - m_kernelSize + 1;
    const size_t outputHeight = m_inputHeight - m_kernelSize + 1;

    Tensor result(outputWidth * outputHeight);
    result.asArray().setZero();

    for(size_t i=0; i<outputWidth; ++i)
      for(size_t j=0; j<outputHeight; ++j)
      {
        const auto inputMatrix  = input.asMatrix(m_inputHeight, m_inputWidth);
        const auto kernelMatrix = kernel.asMatrix(m_kernelSize, m_kernelSize);
        auto resultMatrix = result.asMatrix(outputHeight, outputWidth);

        const auto inputBlock = inputMatrix.block(i, j, m_kernelSize, m_kernelSize);
        resultMatrix(i,j) += inputBlock.cwiseProduct(kernelMatrix).sum();
      }

    return result;
  }

  std::pair<variable_t, variable_t> ConvolutionOperation::gradientsImpl(variable_t gradient, variable_t input, variable_t kernel) const
  {
    const size_t outputWidth  = m_inputWidth - m_kernelSize + 1;
    const size_t outputHeight = m_inputHeight - m_kernelSize + 1;

    // TODO: Support non square input/output, this requires support for
    //       non-square kernel
    assert(outputWidth == outputHeight);
    return std::make_pair(
      std::make_shared<const Variable>(std::vector{gradient, kernel}, std::make_shared<DeconvolutionOperation>(outputWidth, outputHeight, m_kernelSize)),
      std::make_shared<const Variable>(std::vector{input, gradient}, std::make_shared<ConvolutionOperation>(m_inputWidth, m_inputHeight, outputWidth))
    );
  }
}

