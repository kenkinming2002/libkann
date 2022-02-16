#pragma once

#include <libkann/Layer.hpp>

namespace kann
{
  class WeightLayer : public Layer
  {
  public:
    WeightLayer() = default;
    WeightLayer(size_t inputSize, size_t outputSize);

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    std::vector<QualifiedName> parameters(Scope scope) const override;

  public:
    Output process(Scope scope, Input input) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_inputSize, m_outputSize);
    }

  private:
    size_t m_inputSize, m_outputSize;
  };

}

CEREAL_REGISTER_TYPE(kann::WeightLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::WeightLayer);
