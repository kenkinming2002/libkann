#pragma once

#include <libkann/export.hpp>
#include <libkann/Layer.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <vector>
#include <memory>
#include <random>

namespace kann
{
  class NeuralNetworkBase
  {
  public:
    NeuralNetworkBase() = default;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const;
    LIBKANN_SYMEXPORT size_t outputSize() const;

  public:
    LIBKANN_SYMEXPORT void randomize(std::default_random_engine& engine);

  public:
    LIBKANN_SYMEXPORT Eigen::VectorXd feedForward(Eigen::VectorXd input);
    LIBKANN_SYMEXPORT Eigen::RowVectorXd backPropagate(const Eigen::RowVectorXd& outputGradient);
    LIBKANN_SYMEXPORT void train(double learningRate);

  public:
    LIBKANN_SYMEXPORT NeuralNetworkBase cross(const NeuralNetworkBase& other, std::default_random_engine& engine, double mutationRate) const;

  public:
    LIBKANN_SYMEXPORT void addLayer(std::unique_ptr<Layer> layer);

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_layers);
    }

  private:
    std::vector<std::unique_ptr<Layer>> m_layers;
  };
}
