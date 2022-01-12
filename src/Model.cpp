#include <libkann/Model.hpp>

namespace kann
{
  Model::Model(const Model& other)
    : m_layers(other.m_layers)
  {
    for(auto& layer : m_layers)
      layer = layer->clone();
  }

  std::unique_ptr<Model> Model::cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate)
  {
    return std::unique_ptr<Model>(static_cast<Model*>(Layer::cross(lhs, rhs, engine, mutationRate).release()));
  }

  void Model::randomize(std::default_random_engine& engine)
  {
    for(auto& layer : m_layers)
      layer->randomize(engine);
  }

  void Model::train(double learningRate, unsigned tags)
  {
    for(auto& layer : m_layers)
      if(layer->tag() & tags)
        layer->train(learningRate);
      else
        layer->train(0.0); // Clear the gradient
  }

  std::vector<std::span<double>> Model::params()
  {
    std::vector<std::span<double>> params;
    for(auto& layer : m_layers)
    {
      auto paramsLayer = layer->params();
      params.insert(params.end(), paramsLayer.begin(), paramsLayer.end());
    }
    return params;
  }

  std::vector<std::span<const double>> Model::params() const
  {
    std::vector<std::span<const double>> params;
    for(const auto& layer : m_layers)
    {
      auto paramsLayer = layer->params();
      params.insert(params.end(), paramsLayer.begin(), paramsLayer.end());
    }
    return params;
  }


  std::vector<std::span<double>> Model::paramsGradient()
  {
    std::vector<std::span<double>> paramsGradient;
    for(auto& layer : m_layers)
    {
      auto paramsGradientLayer = layer->paramsGradient();
      paramsGradient.insert(paramsGradient.end(), paramsGradientLayer.begin(), paramsGradientLayer.end());
    }
    return paramsGradient;
  }

  std::vector<std::span<const double>> Model::paramsGradient() const
  {
    std::vector<std::span<const double>> paramsGradient;
    for(const auto& layer : m_layers)
    {
      auto paramsGradientLayer = layer->paramsGradient();
      paramsGradient.insert(paramsGradient.end(), paramsGradientLayer.begin(), paramsGradientLayer.end());
    }
    return paramsGradient;
  }

  size_t Model::addLayer(std::shared_ptr<Layer> layer)
  {
    size_t result = m_layers.size();
    m_layers.push_back(std::move(layer));
    return result;
  }

  Layer& Model::layer(size_t index)
  {
    return *m_layers[index];
  }

  const Layer& Model::layer(size_t index) const
  {
    return *m_layers[index];
  }
}

