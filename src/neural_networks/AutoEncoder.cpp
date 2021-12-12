#include <libkann/neural_networks/AutoEncoder.hpp>

#include <libkann/ActivationFunction.hpp>

#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>

namespace kann
{
  size_t AutoEncoder::inputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.front()->inputSize();
  }

  size_t AutoEncoder::outputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.back()->outputSize();
  }

  void AutoEncoder::randomize(std::default_random_engine& engine)
  {
    for(auto& layer : m_layers)
      layer->randomize(engine);
  }

  void AutoEncoder::feedForward(Eigen::VectorXd input)
  {
    m_data.resize(m_layers.size()+1);

    m_data[0] = std::move(input);

    for(size_t i=0; i<m_layers.size(); ++i)
      m_data[i+1] = m_layers[i]->feedForward(m_data[i]);
  }

  void AutoEncoder::backPropagate(const Eigen::VectorXd& expectedOutput)
  {
    // TODO: Allow changing the cost function
    Eigen::VectorXd outputGradient = 2.0 * (output() - expectedOutput);
    for(ssize_t i = m_layers.size()-1; i>=0; --i)
      outputGradient = m_layers[i]->backPropagate(m_data[i], outputGradient);
  }

  void AutoEncoder::train(double learningRate)
  {
    for(auto& layer: m_layers)
      layer->train(learningRate);
  }

  AutoEncoder AutoEncoder::cross(const AutoEncoder& other, std::default_random_engine& engine, double mutationRate) const
  {
    AutoEncoder result;
    for(size_t i=0; i<m_layers.size(); ++i)
      result.addLayer(m_layers[i]->cross(*other.m_layers[i], engine, mutationRate));

    return result;
  }

  void AutoEncoder::addLayer(std::unique_ptr<Layer> layer)
  {
    m_layers.push_back(std::move(layer));
  }

  void AutoEncoder::setFeaturesLayer()
  {
    m_featuresLayerIndex = m_layers.size();
  }

  const Eigen::VectorXd& AutoEncoder::output() const
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

  void AutoEncoder::train(const DataSet& dataSet, float learningRate)
  {
    const size_t size = dataSet.size();

    Eigen::VectorXd input, output;
    for(size_t i = 0; i<size; ++i)
    {
      showProgressBar("Training", i, size);

      dataSet.get(i, input, output);
      this->feedForward(input);
      this->backPropagate(input);
      this->train(learningRate); // TODO: Batching
    }
    std::cout << std::endl;
  }

  void AutoEncoder::generate(Eigen::VectorXd features)
  {
    m_data.resize(m_layers.size()+1);

    m_data[m_featuresLayerIndex] = std::move(features);

    for(size_t i=m_featuresLayerIndex; i<m_layers.size(); ++i)
      m_data[i+1] = m_layers[i]->feedForward(m_data[i]);
  }
}


