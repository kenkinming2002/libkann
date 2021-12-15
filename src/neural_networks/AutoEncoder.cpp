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

  void AutoEncoder::feedForward(Eigen::VectorXd input, FeedForwardResult& result) const
  {
    result.data.resize(m_layers.size()+1);
    result.data.front() = std::move(input);
    for(size_t i=0; i<m_layers.size(); ++i)
      m_layers[i]->feedForward(result.data[i], result.data[i+1]);
  }

  void AutoEncoder::backPropagate(const Eigen::VectorXd& expectedOutput, const FeedForwardResult& feedForwardResult, BackPropagationResult& result) const
  {
    result.gradients.resize(m_layers.size()+1);
    result.layerGradients.resize(m_layers.size());
    result.gradients.back() = 2.0 * (feedForwardResult.output() - expectedOutput);
    for(ssize_t i = m_layers.size()-1; i>=0; --i)
      m_layers[i]->backPropagate(feedForwardResult.data[i], result.gradients[i+1], result.gradients[i], result.layerGradients[i]);
  }

  void AutoEncoder::train(double learningRate, const BackPropagationResult& backPropagationResult)
  {
    for(size_t i=0; i<m_layers.size(); ++i)
      m_layers[i]->train(learningRate, backPropagationResult.layerGradients[i]);
  }

  void AutoEncoder::randomize(std::default_random_engine& engine)
  {
    for(auto& layer : m_layers)
      layer->randomize(engine);
  }

  AutoEncoder AutoEncoder::cross(const AutoEncoder& lhs, const AutoEncoder& rhs, std::default_random_engine& engine, double mutationRate)
  {
    assert(lhs.m_layers.size() == rhs.m_layers.size());

    AutoEncoder result;
    const auto layerSize = lhs.m_layers.size();
    for(size_t i=0; i<layerSize; ++i)
      result.addLayer(Layer::cross(*lhs.m_layers[i], *rhs.m_layers[i], engine, mutationRate));

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
    FeedForwardResult feedForwardResult;
    BackPropagationResult backPropagationResult;
    for(size_t i = 0; i<size; ++i)
    {
      showProgressBar("Training", i, size);

      dataSet.get(i, input, output);
      this->feedForward(input, feedForwardResult);
      this->backPropagate(input, feedForwardResult, backPropagationResult);
      this->train(learningRate, backPropagationResult); // TODO: Batching
    }
    std::cout << std::endl;
  }

  Eigen::VectorXd AutoEncoder::generate(Eigen::VectorXd features)
  {
    Eigen::VectorXd output, input = std::move(features);
    for(size_t i=m_featuresLayerIndex; i<m_layers.size(); ++i)
    {
      m_layers[i]->feedForward(input, output);
      std::swap(input, output);
    }
    return input;
  }
}


