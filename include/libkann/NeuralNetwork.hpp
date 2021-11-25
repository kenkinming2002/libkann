#pragma once

#include <libkann/export.hpp>

#include <libkann/serialization/Eigen.hpp>
#include <libkann/utilities/dynarray.hpp>

#include <libkann/WeightLayer.hpp>
#include <libkann/ActivationLayer.hpp>
#include <libkann/CompositeLayer.hpp>

#include <Eigen/Eigen>

#include <ostream>
#include <vector>
#include <random>
#include <type_traits>
#include <initializer_list>

class NeuralNetwork
{
public:
  NeuralNetwork() = default;

public:
  LIBKANN_SYMEXPORT NeuralNetwork(dynarray<size_t> topology);
  LIBKANN_SYMEXPORT NeuralNetwork(dynarray<size_t> topology, ActivationFunction activationFunction);
  template<typename PRNG>
  LIBKANN_SYMEXPORT NeuralNetwork(dynarray<size_t> topology, PRNG& prng, ActivationFunction activationFunction);

public:
  LIBKANN_SYMEXPORT NeuralNetwork(dynarray<size_t> topology, dynarray<Eigen::MatrixXd> weights, dynarray<ActivationFunction> activationFunctions);

public:
  LIBKANN_SYMEXPORT size_t inputSize() const  { return m_layers.front().inputSize(); }
  LIBKANN_SYMEXPORT size_t outputSize() const { return m_layers.back().outputSize(); }

public:
  const Eigen::VectorXd& output() const { return m_output; }

public:
  LIBKANN_SYMEXPORT void feedForward(Eigen::VectorXd input);
  LIBKANN_SYMEXPORT void backPropagate(const Eigen::VectorXd& expectedOutput);
  LIBKANN_SYMEXPORT void train(double learningRate);

public:
  template<typename PRNG>
  LIBKANN_SYMEXPORT static NeuralNetwork cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, PRNG& prng, double mutationRate);

private:
  using Layer = CompositeLayer<WeightLayer, ActivationLayer>;
  dynarray<Layer> m_layers;

private:
  Eigen::VectorXd m_output;

public:
  template<typename Archive>
  void save(Archive& archive) const
  {
    // topology
    {
      dynarray<size_t> topology(m_layers.size()+1);
      for(size_t i=0; i<m_layers.size(); ++i)
        topology[i] = m_layers[i].inputSize();
      topology.back() = m_output.size();
      archive(topology);
    }

    // weights and activation functions
    {
      dynarray<Eigen::MatrixXd>          weights(m_layers.size());
      dynarray<ActivationFunction::Type> activationFunctionTypes(m_layers.size());
      for(size_t i=0; i<m_layers.size(); ++i)
      {
        weights[i]                 = m_layers[i].layer1().weight();
        activationFunctionTypes[i] = m_layers[i].layer2().activationFunction().type;
      }
      archive(weights);
      archive(activationFunctionTypes);
    }
  }

  template<typename Archive>
  void load(Archive& archive)
  {
    // topology
    dynarray<size_t>                   topology;
    dynarray<Eigen::MatrixXd>          weights;
    dynarray<ActivationFunction::Type> activationFunctionTypes;

    archive(topology);
    archive(weights);
    archive(activationFunctionTypes);

    dynarray<ActivationFunction> activationFunctions(activationFunctionTypes.size());
    for(size_t i=0; i<activationFunctionTypes.size(); ++i)
      activationFunctions[i] = ActivationFunction(activationFunctionTypes[i]);

    *this = NeuralNetwork(std::move(topology), std::move(weights), std::move(activationFunctions));
  }
};
