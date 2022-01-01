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
    LIBKANN_SYMEXPORT void feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const override;
    LIBKANN_SYMEXPORT void backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const override;

  public:
    auto weight() const
    {
      return Eigen::Map<const Eigen::MatrixXd>(params().data(), m_outputSize, m_inputSize);
    }

    auto weightGradient(Eigen::ArrayXd& gradient) const
    {
      return Eigen::Map<Eigen::MatrixXd>(gradient.data(), m_outputSize, m_inputSize);
    }

    auto bias() const
    {
      return Eigen::Map<const Eigen::VectorXd>(params().data()+m_outputSize*m_inputSize, m_outputSize);
    }

    auto biasGradient(Eigen::ArrayXd& gradient) const
    {
      return Eigen::Map<Eigen::VectorXd>(gradient.data()+m_outputSize*m_inputSize, m_outputSize);
    }

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));
      archive(m_inputSize);
      archive(m_outputSize);
    }

  private:
    size_t m_inputSize, m_outputSize;
  };

}

CEREAL_REGISTER_TYPE(kann::WeightLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::WeightLayer);
