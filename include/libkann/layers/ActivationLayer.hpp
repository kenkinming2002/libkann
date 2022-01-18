#pragma once

#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/ActivationFunction.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <Eigen/Eigen>

namespace kann
{
  class ActivationLayer : public Layer
  {
  public:
    LIBKANN_SYMEXPORT ActivationLayer() = default;
    LIBKANN_SYMEXPORT ActivationLayer(size_t size, ActivationFunction activationFunction);

  public:
    LIBKANN_SYMEXPORT std::unique_ptr<Layer> clone() const override;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const override;
    LIBKANN_SYMEXPORT size_t outputSize() const override;

  public:
    LIBKANN_SYMEXPORT std::vector<std::shared_ptr<const Variable>> parametersVariables() const override;
    LIBKANN_SYMEXPORT std::vector<std::shared_ptr<const Tensor>> parameters() const override;
    LIBKANN_SYMEXPORT void parameters(std::vector<std::shared_ptr<const Tensor>> parameters) override;

  public:
    LIBKANN_SYMEXPORT std::pair<std::shared_ptr<const Variable>, StateVariables> operator()(std::shared_ptr<const Variable> input, StateVariables state) const override;

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
