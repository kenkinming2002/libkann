#pragma once

#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/serialization/Eigen.hpp>
#include <libkann/neural_networks/Types.hpp>

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <vector>
#include <memory>
#include <random>

namespace kann
{
  class RecurrentNeuralNetwork
  {
  public:
    LIBKANN_SYMEXPORT RecurrentNeuralNetwork(size_t memory);

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const;
    LIBKANN_SYMEXPORT size_t outputSize() const;

  public:
    LIBKANN_SYMEXPORT void feedForward(Eigen::VectorXd input, RecurrentFeedForwardResult& result) const;

  public:
    LIBKANN_SYMEXPORT void randomize(std::default_random_engine& engine);

  public:
    LIBKANN_SYMEXPORT static RecurrentNeuralNetwork cross(const RecurrentNeuralNetwork& lhs, const RecurrentNeuralNetwork& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    LIBKANN_SYMEXPORT void addLayer(std::unique_ptr<Layer> layer);

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_layers);
      archive(m_memory);
    }

  private:
    std::vector<std::unique_ptr<Layer>> m_layers;
    size_t m_memory;
  };
}
