#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libkann/operations/MultiplyOperation.hpp>
#include <libkann/operations/SubtractOperation.hpp>

namespace kann
{
  SimpleOptimizer::SimpleOptimizer(double learningRate)
    : m_learningRate(learningRate) {}

  SimpleOptimizer::Result SimpleOptimizer::process(VRef parameter, VRef gradient, VMap state) const
  {
    auto scaledGradient = std::make_shared<const Variable>(
      std::vector{std::move(gradient)},
      std::make_shared<MultiplyOperation>(m_learningRate)
    );

    auto newParameter = std::make_shared<const Variable>(
      std::vector{std::move(parameter), std::move(scaledGradient)},
      std::make_shared<SubtractOperation>()
    );

    return Result{
      .newParameter = std::move(newParameter),
      .newState     = std::move(state)
    };
  }
}
