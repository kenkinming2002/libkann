#pragma once

#include <libkann/export.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

#include <Eigen/Eigen>

#include <memory>
#include <random>
#include <utility>
#include <concepts>

namespace kann
{
  class Layer
  {
  public:
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

  public:
    virtual void randomize(std::default_random_engine& engine) = 0;

  public:
    virtual Eigen::VectorXd feedForward(const Eigen::VectorXd& input) = 0;
    virtual Eigen::RowVectorXd backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient) = 0;

  public:
    virtual void train(double learningRate) = 0; // TODO: Improve the interface

  public:
    virtual std::unique_ptr<Layer> cross(const Layer& other, std::default_random_engine& engine, double mutationRate) const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive) {}

  public:
    virtual ~Layer() = default;
  };
}
