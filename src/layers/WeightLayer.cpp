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

  std::vector<Layer::Parameter> WeightLayer::parameters() const
  {
    return {
      Parameter{
        .layer = shared_from_this(),
        .name = "weight",
        .size = m_inputSize * m_outputSize,
        .mean = 0.0,
        .stddev = 1.0 / std::sqrt(m_inputSize)
      },
      Parameter{
        .layer = shared_from_this(),
        .name = "bias",
        .size = m_outputSize,
        .mean = 0.0,
        .stddev = 0.0
      },
    };
  }

  Layer::ProcessOutput WeightLayer::process(ProcessInput input) const
  {
    ProcessOutput output;

    auto weightVariable = input.parameters.at({shared_from_this(), "weight"});
    auto biasVariable   = input.parameters.at({shared_from_this(), "bias"});

    // TODO: Fuse them into a single operation
    auto prodVariable = Variable::apply(MatrixMultiplyOperation(m_outputSize, 1, m_inputSize, false, false), {std::move(weightVariable), std::move(input.variable)});
    output.variable = Variable::apply(ReduceOperation(2), {std::move(biasVariable), std::move(prodVariable)});

    output.states = std::move(input.states);
    return output;
  }

}
