#pragma once

#include <libkann/NewLayer.hpp>

namespace kann
{
  class WeightLayer : public NewLayer
  {
  public:
    WeightLayer() = default;
    WeightLayer(size_t inputSize, size_t outputSize);

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    std::vector<NewParameter> parameters() const override;

  public:
    LayerVariable operator()(Scope scope, LayerVariable input) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<NewLayer>(this));
      archive(m_inputSize, m_outputSize);
    }

  private:
    size_t m_inputSize, m_outputSize;
  };

}

CEREAL_REGISTER_TYPE(kann::WeightLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::NewLayer, kann::WeightLayer);
