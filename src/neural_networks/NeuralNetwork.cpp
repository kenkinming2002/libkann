#include <libkann/neural_networks/NeuralNetwork.hpp>

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>

namespace kann
{
  size_t NeuralNetwork::inputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.front()->inputSize();
  }

  size_t NeuralNetwork::outputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.back()->outputSize();
  }

  void NeuralNetwork::randomize(std::default_random_engine& engine)
  {
    for(auto& layer : m_layers)
      layer->randomize(engine);
  }

  void NeuralNetwork::feedForward(Eigen::VectorXd input)
  {
    m_data.resize(m_layers.size()+1);

    m_data[0] = std::move(input);

    for(size_t i=0; i<m_layers.size(); ++i)
      m_data[i+1] = m_layers[i]->feedForward(m_data[i]);
  }

  void NeuralNetwork::backPropagate(const Eigen::VectorXd& expectedOutput)
  {
    // TODO: Allow changing the cost function
    Eigen::VectorXd outputGradient = 2.0 * (output() - expectedOutput);
    for(ssize_t i = m_layers.size()-1; i>=0; --i)
      outputGradient = m_layers[i]->backPropagate(m_data[i], outputGradient);
  }

  void NeuralNetwork::train(double learningRate)
  {
    for(auto& layer: m_layers)
      layer->train(learningRate);
  }

  NeuralNetwork NeuralNetwork::cross(const NeuralNetwork& other, std::default_random_engine& engine, double mutationRate) const
  {
    NeuralNetwork result;
    for(size_t i=0; i<m_layers.size(); ++i)
      result.addLayer(m_layers[i]->cross(*other.m_layers[i], engine, mutationRate));

    return result;
  }

  void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer)
  {
    m_layers.push_back(std::move(layer));
  }

  const Eigen::VectorXd& NeuralNetwork::output() const
  {
    return m_data.back();
  }

  namespace
  {
    static void showProgressBar(std::string_view name, size_t count, size_t total)
    {
      constexpr static size_t width = 40;
      std::cout << "\e[?25l";

      std::cout << name << "[";
      for(size_t i=0; i<width; ++i)
        if((float)i/width < (float)count/total)
          std::cout << "=";
        else
          std::cout << " ";

      std::cout << " - ";
      std::cout << count << "/" << total;
      std::cout << "]\r";

      std::cout << "\e[?25h";
    }
  }

  void NeuralNetwork::train(const DataSet& dataSet, float learningRate)
  {
    const size_t size = dataSet.size();

    Eigen::VectorXd input, output;
    for(size_t i = 0; i<size; ++i)
    {
      showProgressBar("Training", i, size);

      dataSet.get(i, input, output);
      this->feedForward(input);
      this->backPropagate(output);
      this->train(learningRate); // TODO: Batching
    }
    std::cout << std::endl;
  }

  double NeuralNetwork::test(const DataSet& dataSet)
  {
    const size_t size = dataSet.size();

    Eigen::VectorXd input, output;
    double correctness = 0.0;
    for(size_t i = 0; i<size; ++i)
    {
      showProgressBar("Testing", i, size);

      dataSet.get(i, input, output);
      this->feedForward(input);
      correctness += dataSet.correctness(i, this->output());
    }
    std::cout << std::endl;

    correctness /= size;
    return correctness;
  }
}


