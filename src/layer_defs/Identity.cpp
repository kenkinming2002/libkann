#include <libkann/layer_defs/Identity.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Variable.hpp>

#include <libkann/operations/IdentityOperation.hpp>

namespace kann
{
  YAML::Node IdentityLayerDef::save(layer_def_t layer_def)
  {
    YAML::Node node;
    node["input_size"]  = std::static_pointer_cast<const IdentityLayerDef>(layer_def)->m_input_size;
    node["output_size"] = std::static_pointer_cast<const IdentityLayerDef>(layer_def)->m_output_size;
    node["offset"]      = std::static_pointer_cast<const IdentityLayerDef>(layer_def)->m_offset;
    return node;
  }

  layer_def_t IdentityLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<IdentityLayerDef>();
    layer_def->m_input_size  = node["input_size"].as<size_t>();
    layer_def->m_output_size = node["output_size"].as<size_t>();
    layer_def->m_offset      = node["offset"].as<size_t>();
    return layer_def;
  }

  KANN_LAYER_DEF_SAVE_LOAD_REGISTER(identity, IdentityLayerDef)

  IdentityLayerDef::IdentityLayerDef(size_t input_size, size_t output_size, size_t offset)
    : m_input_size(input_size), m_output_size(output_size), m_offset(offset) {}

  std::shared_ptr<Layer> IdentityLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    return layer;
  }

  size_t IdentityLayerDef::input_size() const
  {
    return m_input_size;
  }

  size_t IdentityLayerDef::output_size() const
  {
    return m_output_size;
  }

  LayerDef::ProcessOutput IdentityLayerDef::process(ProcessInput input) const
  {
    ProcessOutput output;
    output.variable = m_input_size == m_output_size
      ? input.variable
      : Variable::apply(IdentityOperation(m_input_size, m_output_size, m_offset), {input.variable});
    return output;
  }
}

