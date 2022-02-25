#include <libkann/layers/DeconvolutionalLayer.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/DeconvolutionOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

namespace kann
{
  DeconvolutionalLayer::DeconvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
    : m_inputWidth(inputWidth), m_inputHeight(inputHeight),
      m_kernelSize(kernelSize),
      m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount) {}

  size_t DeconvolutionalLayer::inputSize() const
  {
    return m_inputWidth * m_inputHeight * m_inputChannelCount;
  }

  size_t DeconvolutionalLayer::outputSize() const
  {
    return (m_inputWidth+m_kernelSize-1) * (m_inputHeight+m_kernelSize-1) * m_outputChannelCount;
  }

  std::vector<Layer::Parameter> DeconvolutionalLayer::parameters(Scope scope) const
  {
    std::vector<Parameter> results;
    results.reserve(m_inputChannelCount * m_outputChannelCount);

    for(size_t j=0; j<m_outputChannelCount; ++j)
      for(size_t i=0; i<m_inputChannelCount; ++i)
      {
        auto name = QualifiedName{
          .scope = scope,
          .name = "kernel"+std::to_string(i)+"_"+std::to_string(j)
        };

        results.push_back(Parameter{
          .name = name,
          .size = m_kernelSize * m_kernelSize,
          .mean = 0.0,
          .stddev = 1.0 / m_kernelSize
        });
      }

    return results;
  }

  Layer::Output DeconvolutionalLayer::process(Scope scope, Input input) const
  {
    /* TODO: Consider marking them as override final */
    const size_t outputWidth = m_inputWidth + m_kernelSize - 1;
    const size_t outputHeight = m_inputHeight + m_kernelSize - 1;

    // 1: Retrive input variable
    auto inputVariable = std::move(input.input);

    // 2: Split input into channels
    std::vector<CRef<Variable>> inputChannelVariables;
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

    // 3: Perform Deconvolution
    std::vector<CRef<Variable>> resultVariables;
    resultVariables.reserve(m_inputChannelCount * m_outputChannelCount);
    for(size_t j=0; j<m_outputChannelCount; ++j)
      for(size_t i=0; i<m_inputChannelCount; ++i)
      {
        auto inputChannelVariable = inputChannelVariables[i];
        auto kernelParameter = QualifiedName{
          .scope = scope,
          .name = "kernel"+std::to_string(i)+"_"+std::to_string(j)
        };
        auto kernelVariable = input.parameter.at(kernelParameter);
        auto variable = std::make_shared<const Variable>(
          std::vector{std::move(inputChannelVariable), std::move(kernelVariable)},
          std::make_shared<DeconvolutionOperation>(m_inputWidth, m_inputHeight, m_kernelSize)
        );
        resultVariables.push_back(std::move(variable));
      }

    // 4: Reduce result to obtain output channel variables
    std::vector<CRef<Variable>> outputChannelVariables;
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

    return Output{
      std::move(outputVariable),
      std::move(input.inputState)
    };
  }
}
