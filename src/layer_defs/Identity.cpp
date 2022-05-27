#include <libkann/layer_defs/Identity.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Variable.hpp>

#include <libkann/operations/IdentityOperation.hpp>

namespace kann
{
  using namespace std::placeholders;

  IdentityLayerDef::IdentityLayerDef(size_t inputSize, size_t outputSize, size_t offset)
    : m_inputSize(inputSize), m_outputSize(outputSize), m_offset(offset) {}

  std::shared_ptr<Layer> IdentityLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    return layer;
  }

  size_t IdentityLayerDef::input_size() const
  {
    return m_inputSize;
  }

  size_t IdentityLayerDef::output_size() const
  {
    return m_outputSize;
  }

  LayerDef::ProcessOutput IdentityLayerDef::process(ProcessInput input) const
  {
    ProcessOutput output;
    output.variable = m_inputSize == m_outputSize
      ? input.variable
      : Variable::apply(IdentityOperation(m_inputSize, m_outputSize, m_offset), {input.variable});
    return output;
  }
}

