#pragma once

#include <libkann/Layer.hpp>

#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>

namespace kann
{
  class SequentialLayer : public Layer
  {
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
      archive(m_taggedLayers);
    }

  private:
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

CEREAL_REGISTER_TYPE(kann::SequentialLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::SequentialLayer);
