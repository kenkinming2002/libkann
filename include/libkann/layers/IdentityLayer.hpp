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
    LIBKANN_SYMEXPORT IdentityLayer(size_t size);

  public:
    LIBKANN_SYMEXPORT std::unique_ptr<Layer> clone() const override;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const override;
    LIBKANN_SYMEXPORT size_t outputSize() const override;

  public:
    LIBKANN_SYMEXPORT void feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const override;
    LIBKANN_SYMEXPORT void backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_size);
    }

  private:
    size_t m_size;
  };
}

CEREAL_REGISTER_TYPE(kann::IdentityLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::IdentityLayer);

