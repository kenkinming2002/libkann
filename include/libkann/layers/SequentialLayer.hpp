#pragma once

#include <libkann/Layer.hpp>

#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>

namespace kann
{
  class SequentialLayer : public Layer
  {
  public:
    SequentialLayer() = default;
    SequentialLayer(const SequentialLayer& other);

  public:
    void addLayer(std::shared_ptr<Layer> layer, Tag tag = Tag::ALL);

  public:
    std::shared_ptr<Layer> clone() const override;
    void randomize(std::default_random_engine& engine) override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    size_t parameters_count() const override;
    size_t states_count() const override;

    std::vector<size_t> parameter_sizes() const override;
    std::vector<size_t> state_sizes() const override;

    std::vector<std::shared_ptr<const Tensor>> get_parameters() const override;
    std::vector<std::shared_ptr<const Tensor>> get_states() const override;

    void set_parameters(std::vector<std::shared_ptr<const Tensor>> values) override;
    void set_states(std::vector<std::shared_ptr<const Tensor>> values) override;

  public:
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
      std::shared_ptr<Layer> layer;

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
