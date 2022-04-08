#pragma once

#include <libkann/Layer.hpp>

namespace kann
{
  class IdentityLayer : public Layer
  {
  public:
    IdentityLayer() = default;
    IdentityLayer(size_t inputSize, size_t outputSize, size_t offset);

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_inputSize);
      archive(m_outputSize);
      archive(m_offset);
    }

  private:
    size_t m_inputSize;
    size_t m_outputSize;
    size_t m_offset;
  };
}

CEREAL_REGISTER_TYPE(kann::IdentityLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::IdentityLayer);

