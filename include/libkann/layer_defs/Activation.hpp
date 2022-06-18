#pragma once

#include <libkann/Export.hpp>

#include <libkann/LayerDef.hpp>

namespace kann
{
  class KANN_EXPORT ActivationLayerDef : public LayerDef
  {
  public:
    enum class Type
    {
      IDENTITY,
      SIGMOID,
      TANH
    };

  public:
    KANN_EXPORT static YAML::Node save(layer_def_t layer_def);
    KANN_EXPORT static layer_def_t load(YAML::Node node);

  public:
    KANN_EXPORT Shape input_shape() const override;
    KANN_EXPORT Shape output_shape() const override;

  public:
    KANN_EXPORT std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;
    KANN_EXPORT size_t process(Graph& graph, Info& info, size_t input_index) const override;

  private:
    Shape m_shape;
    Type m_type;
  };
}
