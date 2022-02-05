#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libkann/operations/MultiplyOperation.hpp>
#include <libkann/operations/SubtractOperation.hpp>

namespace kann
{
  SimpleOptimizer::SimpleOptimizer(double learningRate)
    : m_learningRate(learningRate) {}

  void SimpleOptimizer::process(Context& context) const
  {
    auto scaledGradient = std::make_shared<const Variable>(
      std::vector{context.gradient},
      std::make_shared<MultiplyOperation>(m_learningRate)
    );

    context.outputParameter = std::make_shared<const Variable>(
      std::vector{context.inputParameter, std::move(scaledGradient)},
      std::make_shared<SubtractOperation>()
    );
  }
}
