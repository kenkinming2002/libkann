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
    void addLayer(CRef<Layer> layer, Tag tag = Tag::ALL);

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    std::vector<Parameter> parameters() const override;
    std::vector<State> states() const override;

    ProcessOutput process(ProcessInput input) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_memory);
      archive(m_taggedLayers);
    }

  private:
    size_t m_memory;

    struct TaggedLayer
    {
      Tag tag;
      CRef<Layer> layer;

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

