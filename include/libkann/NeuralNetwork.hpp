#pragma once

#include <libkann/NeuralNetworkBase.hpp>

namespace kann
{
  class NeuralNetwork : public NeuralNetworkBase
  {
  public:
    using NeuralNetworkBase::NeuralNetworkBase;
    NeuralNetwork(NeuralNetworkBase&& base) : NeuralNetworkBase(std::move(base)) {}

  private:
    using NeuralNetworkBase::feedForward;
    using NeuralNetworkBase::backPropagate;
    using NeuralNetworkBase::cross;
    using NeuralNetworkBase::addLayer;

  public:
    using NeuralNetworkBase::inputSize;
    using NeuralNetworkBase::outputSize;
    using NeuralNetworkBase::train;

  public:
    void feedForward(Eigen::VectorXd input)
    {
      m_output = NeuralNetworkBase::feedForward(std::move(input));
    }

    void backPropagate(Eigen::VectorXd expectedOutput)
    {
      Eigen::RowVectorXd outputGradient = 2.0 * (m_output - expectedOutput).transpose();
      NeuralNetworkBase::backPropagate(outputGradient);
    }

  public:
    NeuralNetwork cross(const NeuralNetwork& other, std::default_random_engine& engine, double mutationRate) const
    {
      return NeuralNetworkBase::cross(other, engine, mutationRate);
    }

  public:
    void addLayer(std::unique_ptr<Layer> layer)
    {
      NeuralNetworkBase::addLayer(std::move(layer));
      m_output = Eigen::VectorXd::Zero(outputSize());
    }

  public:
    const Eigen::VectorXd& output() const { return m_output; }

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<NeuralNetworkBase>(this));
      archive(m_output); // Do we really want to save this?
    }

  private:
    Eigen::VectorXd m_output;
  };
}
