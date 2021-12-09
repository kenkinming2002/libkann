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
    void weight(Eigen::MatrixXd weight)
    {
      assert(m_weight.rows() == weight.rows());
      assert(m_weight.cols() == weight.cols());
      m_weight = weight;
    }

    Eigen::MatrixXd weight() const
    {
      return m_weight;
    }

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));

      archive(m_weight);
      archive(m_weightGradient);
    }

  private:
    Eigen::MatrixXd m_weight;
    Eigen::MatrixXd m_weightGradient;
  };

}

CEREAL_REGISTER_TYPE(kann::WeightLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::WeightLayer);
