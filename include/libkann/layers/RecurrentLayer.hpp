#pragma once

#include <libkann/Layer.hpp>

#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>

namespace kann
{
  class RecurrentLayer : public Layer
  {
  public:
    RecurrentLayer() = default;
    RecurrentLayer(size_t memory);

  public:
    void addLayer(std::shared_ptr<const Layer> layer, Tag tag = Tag::ALL);

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
      archive(m_taggedLayers);
    }

  private:
    Scope layerScope(size_t i) const
    {
      return Scope(m_taggedLayers[i].tag, "layer"+std::to_string(i));
    }

  private:
    size_t m_memory;

    struct TaggedLayer
    {
      std::shared_ptr<const Layer> layer;
      Tag tag;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(layer, tag);
      }
    };
    std::vector<TaggedLayer> m_taggedLayers;
  };
}

CEREAL_REGISTER_TYPE(kann::RecurrentLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::RecurrentLayer);

