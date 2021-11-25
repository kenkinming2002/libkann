#include <libkann/NeuralNetwork.hpp>

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>

NeuralNetwork::NeuralNetwork(dynarray<size_t> topology)
{
  m_layers = dynarray<Layer>(topology.size()-1);
  for(size_t i=0; i<topology.size()-1; ++i)
  {
    size_t inputSize  = topology[i];
    size_t outputSize = topology[i+1];
    auto weightLayer     = WeightLayer(inputSize, outputSize);
    auto activationLayer = ActivationLayer(outputSize);
    m_layers[i] = Layer(std::move(weightLayer), std::move(activationLayer));
  }
  m_output = Eigen::VectorXd(topology.back());
}

NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, ActivationFunction activationFunction)
  : NeuralNetwork(std::move(topology))
{
  for(auto& layer : m_layers)
    layer.layer2().activationFunction(activationFunction);
}

template<typename PRNG>
NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, PRNG& prng, ActivationFunction activationFunction)
  : NeuralNetwork(std::move(topology), activationFunction)
{
  for(auto& layer : m_layers)
    layer.layer1().randomize(prng);
}
template NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, std::mt19937& prng, ActivationFunction activationFunction);

NeuralNetwork::NeuralNetwork(dynarray<size_t> topology, dynarray<Eigen::MatrixXd> weights, dynarray<ActivationFunction> activationFunctions)
  : NeuralNetwork(topology)
{
  for(size_t i=0; i<m_layers.size(); ++i)
  {
    m_layers[i].layer1().weight(std::move(weights[i]));
    m_layers[i].layer2().activationFunction(activationFunctions[i]);
  }
}

void NeuralNetwork::feedForward(Eigen::VectorXd input)
{
  for(auto& layer : m_layers)
    input = layer.feedForward(input);

  m_output = input;
}

void NeuralNetwork::backPropagate(const Eigen::VectorXd& expectedOutput)
{
  Eigen::VectorXd outputGradient = 2.0 * (m_output - expectedOutput);
  for(auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
  {
    auto& layer = *it;
    outputGradient = layer.backPropagate(outputGradient);
  }
}

void NeuralNetwork::train(double learningRate)
{
  for(auto& layer: m_layers)
    layer.layer1().train(learningRate);
}

template<typename PRNG>
NeuralNetwork NeuralNetwork::cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, PRNG& prng, double mutationRate)
{
  NeuralNetwork result = lhs;
  for(size_t i=0; i<result.m_layers.size(); ++i)
    result.m_layers[i].layer1() = WeightLayer::cross(
      lhs.m_layers[i].layer1(),
      rhs.m_layers[i].layer1(),
      prng, mutationRate
    );
  return result;
}

template NeuralNetwork NeuralNetwork::cross<std::mt19937>(const NeuralNetwork& lhs, const NeuralNetwork& rhs, std::mt19937& prng, double mutationRate);

