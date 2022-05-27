#include <libkann/layer_defs/Activation.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Variable.hpp>

#include <libkann/operations/CWiseOperation.hpp>

namespace kann
{
  ActivationLayerDef::ActivationLayerDef(size_t size, ActivationFunction activationFunction)
    : m_size(size), m_activationFunction(activationFunction) {}

  std::shared_ptr<Layer> ActivationLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    return layer;
  }

  size_t ActivationLayerDef::input_size() const
  {
    return m_size;
  }

  size_t ActivationLayerDef::output_size() const
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

  LayerDef::ProcessOutput ActivationLayerDef::process(ProcessInput input) const
  {
    ProcessOutput output;
    output.variable = Variable::apply(ActivationOperation(m_activationFunction), {input.variable});
    output.states = input.states;
    return output;
  }
}
