#include <libkann/layers/ConvolutionalLayer.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/ConvolutionOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

namespace kann
{
  ConvolutionalLayer::ConvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
    : m_inputWidth(inputWidth), m_inputHeight(inputHeight),
      m_kernelSize(kernelSize),
      m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount) {}

  size_t ConvolutionalLayer::inputSize() const
  {
    return m_inputWidth * m_inputHeight * m_inputChannelCount;
  }

  size_t ConvolutionalLayer::outputSize() const
  {
    return (m_inputWidth-m_kernelSize+1) * (m_inputHeight-m_kernelSize+1) * m_outputChannelCount;
  }

  std::vector<QualifiedName> ConvolutionalLayer::parameters(Scope scope) const
  {
    std::vector<QualifiedName> results;
    results.reserve(m_inputChannelCount * m_outputChannelCount);

    for(size_t j=0; j<m_outputChannelCount; ++j)
      for(size_t i=0; i<m_inputChannelCount; ++i)
        results.push_back(QualifiedName{
          .scope = scope,
          .name = "kernel"+std::to_string(i)+"_"+std::to_string(j),
          .size = m_kernelSize * m_kernelSize
        });

    return results;
  }

  LayerVariable ConvolutionalLayer::operator()(Scope scope, LayerVariable input) const
  {
    /* TODO: Consider marking them as override final */
    const size_t outputWidth = m_inputWidth - m_kernelSize + 1;
    const size_t outputHeight = m_inputHeight - m_kernelSize + 1;

    // 1: Retrive input variable
    auto inputVariable = input.variable;

    // 2: Split input into channels
    std::vector<std::shared_ptr<const Variable>> inputChannelVariables;
    inputChannelVariables.reserve(m_inputChannelCount);
    for(size_t i=0; i<m_inputChannelCount; ++i)
    {
      auto variable = std::make_shared<const Variable>(
        std::vector{inputVariable},
        std::make_shared<IdentityOperation>(
          m_inputWidth * m_inputHeight * m_inputChannelCount,
          m_inputWidth * m_inputHeight,
          m_inputWidth * m_inputHeight * i
        )
      );
      inputChannelVariables.push_back(std::move(variable));
    }

    // 3: Perform Convolution
    std::vector<std::shared_ptr<const Variable>> resultVariables;
    resultVariables.reserve(m_inputChannelCount * m_outputChannelCount);
    for(size_t j=0; j<m_outputChannelCount; ++j)
      for(size_t i=0; i<m_inputChannelCount; ++i)
      {
        auto inputChannelVariable = inputChannelVariables[i];
        auto kernelParameter = QualifiedName{
          .scope = scope,
          .name = "kernel"+std::to_string(i)+"_"+std::to_string(j),
          .size = m_kernelSize * m_kernelSize
        };
        auto kernelVariable = input.lookup(LayerVariable::Type::PARAMETER, kernelParameter);
        auto variable = std::make_shared<const Variable>(
          std::vector{std::move(inputChannelVariable), std::move(kernelVariable)},
          std::make_shared<ConvolutionOperation>(m_inputWidth, m_inputHeight, m_kernelSize)
        );
        resultVariables.push_back(std::move(variable));
      }

    // 4: Reduce result to obtain output channel variables
    std::vector<std::shared_ptr<const Variable>> outputChannelVariables;
    for(size_t j=0; j<m_outputChannelCount; ++j)
    {
      auto tmp = std::vector(
        &resultVariables[m_inputChannelCount * j],
        &resultVariables[m_inputChannelCount * (j+1)]
      );
      auto variable = std::make_shared<const Variable>(
        std::move(tmp),
        std::make_shared<ReduceOperation>(m_inputChannelCount)
      );
      outputChannelVariables.push_back(std::move(variable));
    }

    // 5: Concat output channel variables
    for(size_t i=0; i<m_outputChannelCount; ++i)
      outputChannelVariables[i] = std::make_shared<const Variable>(
        std::vector{std::move(outputChannelVariables[i])},
        std::make_shared<IdentityOperation>(
          outputWidth * outputHeight,
          outputWidth * outputHeight * m_outputChannelCount,
          outputWidth * outputHeight * i
        )
      );

    auto outputVariable = std::make_shared<const Variable>(
      std::move(outputChannelVariables),
      std::make_shared<ReduceOperation>(m_outputChannelCount)
    );

    auto output = std::move(input);
    output.variable = std::move(outputVariable);
    return output;
  }
}
