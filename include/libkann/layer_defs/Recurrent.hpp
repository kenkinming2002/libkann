#pragma once

#include <libkann/layer_defs/Sequential.hpp>

namespace kann
{
  class RecurrentLayerDef : public SequentialLayerDef
  {
  public:
    static YAML::Node save(layer_def_t layer_def);
    static layer_def_t load(YAML::Node node);

  public:
    RecurrentLayerDef() = default;
    RecurrentLayerDef(size_t memory_size);

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  protected:
    size_t states_count() const override;
    std::vector<size_t> states_sizes() const override;

  private:
    size_t m_memory_size;
  };
}

