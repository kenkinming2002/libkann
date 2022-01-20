#include <libkann/layers/ActivationLayer.hpp>

#include <libkann/operations/CWiseOperation.hpp>

#include <functional>

namespace kann
{
  using namespace std::placeholders;

  ActivationLayer::ActivationLayer(size_t size, ActivationFunction activationFunction)
    : m_size(size), m_activationFunction(activationFunction) {}

  std::unique_ptr<Layer> ActivationLayer::clone() const
  {
    return std::make_unique<ActivationLayer>(*this);
  }

  size_t ActivationLayer::inputSize() const
  {
    return m_size;
  }

  size_t ActivationLayer::outputSize() const
  {
    return m_size;
  }

  std::vector<std::shared_ptr<const Variable>> ActivationLayer::parametersVariables(unsigned tags) const
  {
    return {};
  }

  std::vector<std::reference_wrapper<const std::shared_ptr<const Tensor>>> ActivationLayer::parameters(unsigned tags) const
  {
    return {};
  }

  std::vector<std::reference_wrapper<std::shared_ptr<const Tensor>>> ActivationLayer::parameters(unsigned tags)
  {
    return {};
  }

  auto ActivationLayer::operator()(std::shared_ptr<const Variable> input, StateVariables state) const -> std::pair<std::shared_ptr<const Variable>, StateVariables>
  {
    const auto activationFunction = m_activationFunction;
    const auto normal     = [=](double val){ return activationFunction.normal(val); };
    const auto derivative = [=](double val){ return activationFunction.derivative(val); };

    auto output = std::make_shared<const Variable>(
      std::vector{std::move(input)},
      std::make_shared<CWiseOperation<decltype(normal), decltype(derivative)>>(normal, derivative)
    );

    return std::make_pair(std::move(output), std::move(state));
  }
}
