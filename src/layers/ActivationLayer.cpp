#include <libkann/layers/ActivationLayer.hpp>

#include <libkann/operations/CWiseOperation.hpp>

namespace kann
{
  using namespace std::placeholders;

  ActivationLayer::ActivationLayer(size_t size, ActivationFunction activationFunction)
    : m_size(size), m_activationFunction(activationFunction) {}

  size_t ActivationLayer::inputSize() const
  {
    return m_size;
  }

  size_t ActivationLayer::outputSize() const
  {
    return m_size;
  }

  LayerVariable ActivationLayer::operator()(Scope scope, LayerVariable input) const
  {
    const auto activationFunction = m_activationFunction;
    const auto normal     = [=](double val){ return activationFunction.normal(val); };
    const auto derivative = [=](double val){ return activationFunction.derivative(val); };

    auto output = std::move(input);
    output.variable = std::make_shared<const Variable>(
      std::vector{std::move(output.variable)},
      std::make_shared<CWiseOperation<decltype(normal), decltype(derivative)>>(normal, derivative)
    );
    return output;
  }
}
