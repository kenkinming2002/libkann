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
    std::vector<State> states(Scope scope) const override;

    Output process(Scope scope, Input input) const override;

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
      return Scope("layer"+std::to_string(i), m_taggedLayers[i].tag);
    }

  private:
    size_t m_memory;

    struct TaggedLayer
    {
      Tag tag;
      std::shared_ptr<const Layer> layer;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(tag, layer);
      }
    };
    std::vector<TaggedLayer> m_taggedLayers;
  };
}

CEREAL_REGISTER_TYPE(kann::RecurrentLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::RecurrentLayer);

