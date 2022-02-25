#include <libkann/operations/DeconvolutionOperation.hpp>

#include <libkann/operations/ConvolutionOperation.hpp>

#include <libkann/Variable.hpp>

namespace kann
{
  DeconvolutionOperation::DeconvolutionOperation(size_t inputWidth, size_t inputHeight, size_t kernelSize)
    : m_inputWidth(inputWidth), m_inputHeight(inputHeight), m_kernelSize(kernelSize) {}

  Tensor DeconvolutionOperation::processImpl(const Tensor& input, const Tensor& kernel) const
  {
    const size_t outputWidth  = m_inputWidth + m_kernelSize - 1;
    const size_t outputHeight = m_inputHeight + m_kernelSize - 1;

    Tensor result(outputWidth * outputHeight);
    result.asArray().setZero();

    for(size_t i=0; i<m_inputWidth; ++i)
      for(size_t j=0; j<m_inputHeight; ++j)
      {
        const auto inputMatrix  = input.asMatrix(m_inputHeight, m_inputWidth);
        const auto kernelMatrix = kernel.asMatrix(m_kernelSize, m_kernelSize);
        auto resultMatrix = result.asMatrix(outputHeight, outputWidth);

        auto resultBlock = resultMatrix.block(i, j, m_kernelSize, m_kernelSize);
        resultBlock += inputMatrix(i,j) * kernelMatrix;
      }

    return result;
  }

  std::pair<CRef<Variable>, CRef<Variable>> DeconvolutionOperation::gradientsImpl(CRef<Variable> gradient, CRef<Variable> input, CRef<Variable> kernel) const
  {
    const size_t outputWidth  = m_inputWidth + m_kernelSize - 1;
    const size_t outputHeight = m_inputHeight + m_kernelSize - 1;

    // TODO: Support non square input/output, this requires support for
    //       non-square kernel
    assert(m_inputWidth == m_inputHeight);
    return std::make_pair(
      std::make_shared<const Variable>(std::vector{gradient, kernel}, std::make_shared<ConvolutionOperation>(outputWidth, outputHeight, m_kernelSize)),
      std::make_shared<const Variable>(std::vector{gradient, input}, std::make_shared<ConvolutionOperation>(outputWidth, outputHeight, m_inputWidth))
    );
  }
}


