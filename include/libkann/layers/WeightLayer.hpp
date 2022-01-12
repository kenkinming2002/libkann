#pragma once

#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <Eigen/Eigen>

#include <random>
#include <assert.h>

namespace kann
{
  class WeightLayer : public Layer
  {
  public:
    LIBKANN_SYMEXPORT WeightLayer() = default;
    LIBKANN_SYMEXPORT WeightLayer(size_t inputSize, size_t outputSize);

  public:
    LIBKANN_SYMEXPORT std::unique_ptr<Layer> clone() const override;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const override;
    LIBKANN_SYMEXPORT size_t outputSize() const override;

  public:
    LIBKANN_SYMEXPORT Eigen::VectorXd feedForward() override;
    LIBKANN_SYMEXPORT Eigen::VectorXd backPropagate() override;

  protected:
    LIBKANN_SYMEXPORT std::vector<std::span<double>> params() override;
    LIBKANN_SYMEXPORT std::vector<std::span<const double>> params() const override;

    LIBKANN_SYMEXPORT std::vector<std::span<double>> paramsGradient() override;
    LIBKANN_SYMEXPORT std::vector<std::span<const double>> paramsGradient() const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_inputSize);
      archive(m_outputSize);
      archive(m_weight, m_weightGradient);
      archive(m_bias, m_biasGradient);
    }

  private:
    size_t m_inputSize, m_outputSize;

  private:
    Eigen::MatrixXd m_weight;
    Eigen::MatrixXd m_weightGradient;

    Eigen::VectorXd m_bias;
    Eigen::VectorXd m_biasGradient;
  };

}

CEREAL_REGISTER_TYPE(kann::WeightLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::WeightLayer);
