#pragma once

#include <libkann/export.hpp>
#include <libkann/Layer.hpp>
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
    LIBKANN_SYMEXPORT size_t inputSize() const override;
    LIBKANN_SYMEXPORT size_t outputSize() const override;

  public:
    LIBKANN_SYMEXPORT void randomize(std::default_random_engine& engine) override;

  public:
    LIBKANN_SYMEXPORT Eigen::VectorXd feedForward() override;
    LIBKANN_SYMEXPORT Eigen::RowVectorXd backPropagate(const Eigen::RowVectorXd& outputGradient) override;
    LIBKANN_SYMEXPORT void train(double learningRate) override;

  public:
    LIBKANN_SYMEXPORT std::unique_ptr<Layer> cross(const Layer& other, std::default_random_engine& engine, double mutationRate) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));

      archive(m_activationFunction);
    }

  private:
    size_t m_size;
    ActivationFunction m_activationFunction;
  };
}

CEREAL_REGISTER_TYPE(kann::ActivationLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::ActivationLayer);
