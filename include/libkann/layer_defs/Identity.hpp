#pragma once

#include <libkann/LayerDef.hpp>

namespace kann
{
  class IdentityLayerDef : public LayerDef
  {
  public:
    static YAML::Node save(layer_def_t layer_def);
    static layer_def_t load(YAML::Node node);

  public:
    IdentityLayerDef() = default;
    IdentityLayerDef(size_t input_size, size_t output_size, size_t offset);

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  private:
    size_t m_input_size;
    size_t m_output_size;
    size_t m_offset;
  };
}

