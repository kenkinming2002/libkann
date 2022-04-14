#include <libkann/layers/ActivationLayer.hpp>

#include <libkann/operations/CWiseOperation.hpp>

namespace kann
{
  ActivationLayer::ActivationLayer(size_t size, ActivationFunction activationFunction)
    : m_size(size), m_activationFunction(activationFunction) {}

  std::shared_ptr<Layer> ActivationLayer::clone() const
  {
    return std::make_shared<ActivationLayer>(*this);
  }

  size_t ActivationLayer::input_size() const { return m_size; }
  size_t ActivationLayer::output_size() const { return m_size; }

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

  Layer::ProcessOutput ActivationLayer::process(ProcessInput input) const
  {
    ProcessOutput output;
    output.variable = Variable::apply(ActivationOperation(m_activationFunction), {input.variable});
    output.states = input.states;
    return output;
  }
}
