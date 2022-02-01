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

  std::vector<NewParameter> WeightLayer::parameters() const
  {
    auto weightParameter = NewParameter{
      .name = "weight",
      .size = m_inputSize * m_outputSize
    };

    auto biasParameter = NewParameter{
      .name = "bias",
      .size = m_outputSize
    };

    return {weightParameter, biasParameter};
  }

  LayerVariable WeightLayer::operator()(Scope scope, LayerVariable input) const
  {
    auto inputVariable = std::move(input.variable);

    auto weightParameter = NewParameter{
      .scope = scope,
      .name = "weight",
      .size = m_inputSize * m_outputSize
    };

    auto biasParameter = NewParameter{
      .scope = scope,
      .name = "bias",
      .size = m_outputSize
    };

    auto weightVariable = input.lookup(LayerVariable::Type::PARAMETER, weightParameter);
    auto biasVariable   = input.lookup(LayerVariable::Type::PARAMETER, biasParameter);

    // TODO: Fuse them into a single operation
    auto prodVariable = std::make_shared<const Variable>(
      std::vector{std::move(weightVariable), std::move(inputVariable)},
      std::make_shared<MatrixMultiplyOperation>(m_outputSize, 1, m_inputSize, false, false)
    );

    auto outputVariable = std::make_shared<const Variable>(
      std::vector{std::move(biasVariable), std::move(prodVariable)},
      std::make_shared<ReduceOperation>(2)
    );

    auto output = std::move(input);
    output.variable = std::move(outputVariable);
    return output;
  }

}
