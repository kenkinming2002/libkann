#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libkann/Variable.hpp>

#include <libkann/operations/MultiplyOperation.hpp>
#include <libkann/operations/SubtractOperation.hpp>

namespace kann
{
  SimpleOptimizer::SimpleOptimizer(double learningRate)
    : m_learningRate(learningRate) {}

  Optimizer::ProcessOutput SimpleOptimizer::process(ProcessInput input) const
  {
    ProcessOutput output;

    auto scaledGradient = std::make_shared<const Variable>(
      std::vector{input.gradient},
      std::make_shared<MultiplyOperation>(m_learningRate)
    );

    output.parameter = std::make_shared<const Variable>(
      std::vector{input.parameter, std::move(scaledGradient)},
      std::make_shared<SubtractOperation>()
    );

    return output;
  }
}
