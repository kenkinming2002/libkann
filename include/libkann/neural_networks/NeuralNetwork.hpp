#pragma once

#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/datasets/DataSet.hpp>
#include <libkann/serialization/Eigen.hpp>
#include <libkann/neural_networks/Types.hpp>

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <vector>
#include <memory>
#include <random>

namespace kann
{
  class NeuralNetwork
  {
  public:
    NeuralNetwork() = default;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const;
    LIBKANN_SYMEXPORT size_t outputSize() const;

  public:
    LIBKANN_SYMEXPORT void feedForward(Eigen::VectorXd input, FeedForwardResult& result) const;
    LIBKANN_SYMEXPORT void backPropagate(const Eigen::VectorXd& expectedOutput, const FeedForwardResult& feedForwardResult, BackPropagationResult& result) const;
    LIBKANN_SYMEXPORT void train(double learningRate, const BackPropagationResult& backPropagationResult);

  public:
    LIBKANN_SYMEXPORT void randomize(std::default_random_engine& engine);

  public:
    LIBKANN_SYMEXPORT static NeuralNetwork cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    LIBKANN_SYMEXPORT void addLayer(std::unique_ptr<Layer> layer);

  public:
    LIBKANN_SYMEXPORT void train(const DataSet& dataSet, float learningRate);
    LIBKANN_SYMEXPORT double test(const DataSet& dataSet) const;

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
