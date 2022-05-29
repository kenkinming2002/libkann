#pragma once

#include <libkann/LayerDef.hpp>

namespace kann
{
  class SequentialLayerDef : public LayerDef
  {
  public:
    static YAML::Node save(layer_def_t layer_def);
    static layer_def_t load(YAML::Node node);

  public:
    SequentialLayerDef() = default;

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;
  };
}
