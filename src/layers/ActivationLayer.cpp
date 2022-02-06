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

  class ActivationOperation : public CWiseOperation<ActivationOperation, 1>
  {
  public:
    constexpr ActivationOperation(ActivationFunction activationFunction)
      : m_activationFunction(activationFunction) {}

  public:
    double forward(double input) const
    {
      return m_activationFunction.normal(input);
    }

    double backward(size_t index, double gradient, double input) const
    {
      assert(index == 0);
      return gradient * m_activationFunction.derivative(input);
    }

  private:
    ActivationFunction m_activationFunction;
  };

  LayerVariable ActivationLayer::operator()(Scope scope, LayerVariable input) const
  {
    auto output = std::move(input);
    output.variable = std::make_shared<const Variable>(
      std::vector{std::move(output.variable)},
      std::make_shared<ActivationOperation>(m_activationFunction)
    );
    return output;
  }
}
