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
    LIBKANN_SYMEXPORT void feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const override;
    LIBKANN_SYMEXPORT void backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const override;

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
