#pragma once

#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/datasets/DataSet.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <vector>
#include <memory>
#include <random>

namespace kann
{
  class AutoEncoder
  {
  public:
    AutoEncoder() = default;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const;
    LIBKANN_SYMEXPORT size_t outputSize() const;

  public:
    LIBKANN_SYMEXPORT void randomize(std::default_random_engine& engine);

  public:
    LIBKANN_SYMEXPORT void feedForward(Eigen::VectorXd input);
    LIBKANN_SYMEXPORT void backPropagate(const Eigen::VectorXd& expectedOutput);
    LIBKANN_SYMEXPORT void train(double learningRate);

  public:
    LIBKANN_SYMEXPORT AutoEncoder cross(const AutoEncoder& other, std::default_random_engine& engine, double mutationRate) const;

  public:
    LIBKANN_SYMEXPORT void addLayer(std::unique_ptr<Layer> layer);
    LIBKANN_SYMEXPORT void setFeaturesLayer();
    LIBKANN_SYMEXPORT const Eigen::VectorXd& output() const;

  public:
    LIBKANN_SYMEXPORT void train(const DataSet& dataSet, float learningRate);
    LIBKANN_SYMEXPORT void generate(Eigen::VectorXd features);

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_featuresLayerIndex);
      archive(m_layers);
      archive(m_data);
    }

  private:
    size_t m_featuresLayerIndex = 0;
    std::vector<std::unique_ptr<Layer>> m_layers;
    std::vector<Eigen::VectorXd> m_data;
  };
}
