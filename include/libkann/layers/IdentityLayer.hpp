#pragma once

#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/ActivationFunction.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <Eigen/Eigen>

namespace kann
{
  class IdentityLayer : public Layer
  {
  public:
    LIBKANN_SYMEXPORT IdentityLayer() = default;
    LIBKANN_SYMEXPORT IdentityLayer(size_t inputSize, size_t outputSize, size_t offset);

  public:
    LIBKANN_SYMEXPORT std::unique_ptr<Layer> clone() const override;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const override;
    LIBKANN_SYMEXPORT size_t outputSize() const override;

  public:
    LIBKANN_SYMEXPORT std::vector<std::shared_ptr<const Variable>> parametersVariables() const override;
    LIBKANN_SYMEXPORT std::vector<std::reference_wrapper<const Tensor>> parameters() const override;
    LIBKANN_SYMEXPORT std::vector<std::reference_wrapper<Tensor>> parameters() override;

  public:
    LIBKANN_SYMEXPORT std::pair<std::shared_ptr<const Variable>, StateVariables> operator()(std::shared_ptr<const Variable> input, StateVariables state) const override;

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

