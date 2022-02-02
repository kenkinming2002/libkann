#pragma once

#include <libkann/Layer.hpp>

namespace kann
{
  class RecurrentLayer : public Layer
  {
  public:
    RecurrentLayer() = default;
    RecurrentLayer(size_t memory);

  public:
    void addLayer(std::shared_ptr<const Layer> layer);

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    std::vector<Parameter> parameters(Scope scope) const override;
    std::vector<Parameter> stateParameters(Scope scope) const override;

    LayerVariable operator()(Scope scope, LayerVariable) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_memory);
      archive(m_layers);
    }

  private:
    static Scope layerScope(size_t i)
    {
      return Scope("layer"+std::to_string(i));
    }

  private:
    size_t m_memory;
    std::vector<std::shared_ptr<const Layer>> m_layers;
  };
}

CEREAL_REGISTER_TYPE(kann::RecurrentLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::RecurrentLayer);

