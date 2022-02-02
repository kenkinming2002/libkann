#pragma once

#include <libkann/Layer.hpp>

namespace kann
{
  class SequentialLayer : public Layer
  {
  public:
    void addLayer(std::shared_ptr<const Layer> layer);

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    std::vector<Parameter> parameters() const override;
    std::vector<Parameter> stateParameters() const override;

    LayerVariable operator()(Scope scope, LayerVariable) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_layers);
    }

  private:
    static Scope layerScope(size_t i)
    {
      return Scope("layer"+std::to_string(i));
    }

  private:
    std::vector<std::shared_ptr<const Layer>> m_layers;
  };
}

CEREAL_REGISTER_TYPE(kann::SequentialLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::SequentialLayer);
