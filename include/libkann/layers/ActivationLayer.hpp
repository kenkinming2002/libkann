#pragma once

#include <libkann/Layer.hpp>
#include <libkann/ActivationFunction.hpp>

namespace kann
{
  class ActivationLayer : public Layer
  {
  public:
    ActivationLayer() = default;
    ActivationLayer(size_t size, ActivationFunction activationFunction);

  public:
    std::shared_ptr<Layer> clone() const override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_size);
      archive(m_activationFunction);
    }

  private:
    size_t m_size;
    ActivationFunction m_activationFunction;
  };
}

CEREAL_REGISTER_TYPE(kann::ActivationLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::ActivationLayer);
