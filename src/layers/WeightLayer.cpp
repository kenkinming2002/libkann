#include <libkann/layers/WeightLayer.hpp>

#include <libkann/operations/MatrixMultiplyOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

namespace kann
{
  WeightLayer::WeightLayer(size_t inputSize, size_t outputSize)
    : m_inputSize(inputSize), m_outputSize(outputSize) {}

  size_t WeightLayer::inputSize() const
  {
    return m_inputSize;
  }

  size_t WeightLayer::outputSize() const
  {
    return m_outputSize;
  }

  std::vector<QualifiedName> WeightLayer::parameters(Scope scope) const
  {
    auto weightParameter = QualifiedName{
      .scope = scope,
      .name = "weight",
      .size = m_inputSize * m_outputSize,
      .mean = 0.0,
      .stddev = 1.0 / std::sqrt(m_inputSize)
    };

    auto biasParameter = QualifiedName{
      .scope = scope,
      .name = "bias",
      .size = m_outputSize,
      .mean = 0.0,
      .stddev = 0.0
    };

    return {weightParameter, biasParameter};
  }

  Layer::Output WeightLayer::process(Scope scope, Input input) const
  {
    auto inputVariable = std::move(input.input);

    auto weightParameter = QualifiedName{
      .scope = scope,
      .name = "weight",
      .size = m_inputSize * m_outputSize,
      .mean = 0.0,
      .stddev = 1.0 / std::sqrt(m_inputSize)
    };

    auto biasParameter = QualifiedName{
      .scope = scope,
      .name = "bias",
      .size = m_outputSize,
      .mean = 0.0,
      .stddev = 0.0
    };

    auto weightVariable = input.parameter.at(weightParameter);
    auto biasVariable   = input.parameter.at(biasParameter);

    // TODO: Fuse them into a single operation
    auto prodVariable = std::make_shared<const Variable>(
      std::vector{std::move(weightVariable), std::move(inputVariable)},
      std::make_shared<MatrixMultiplyOperation>(m_outputSize, 1, m_inputSize, false, false)
    );

    auto outputVariable = std::make_shared<const Variable>(
      std::vector{std::move(biasVariable), std::move(prodVariable)},
      std::make_shared<ReduceOperation>(2)
    );

    return Output{
      std::move(outputVariable),
      std::move(input.inputState)
    };
  }

}
