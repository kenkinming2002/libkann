#include <libkann/Model.hpp>

namespace kann
{
  Model::Model(const Model& other)
    : m_layers(other.m_layers)
  {
    for(auto& layer : m_layers)
      layer = layer->clone();
  }

  std::vector<std::shared_ptr<const Variable>> Model::parametersVariables() const
  {
    std::vector<std::shared_ptr<const Variable>> result;
    for(const auto& layer : m_layers)
    {
      auto parameterVariables = layer->parametersVariables();
      result.insert(result.end(), parameterVariables.begin(), parameterVariables.end());
    }
    return result;
  }

  std::vector<std::shared_ptr<const Tensor>> Model::parameters() const
  {
    std::vector<std::shared_ptr<const Tensor>> parameters;
    for(const auto& layer : m_layers)
    {
      auto layerParameters = layer->parameters();
      parameters.insert(parameters.end(), layerParameters.begin(), layerParameters.end());
    }
    return parameters;
  }

  void Model::parameters(std::vector<std::shared_ptr<const Tensor>> parameters)
  {
    auto it = parameters.begin();
    for(const auto& layer : m_layers)
    {
      // TODO: consider having a virtual function that returns the number of
      //       parameters
      size_t count = layer->parameters().size();
      auto end = std::next(it, count);
      auto layerParameters = std::vector(it, end);
      layer->parameters(std::move(layerParameters));

      it = end;
    }
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

  std::unique_ptr<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate)
  {
    std::unique_ptr<Layer> result = cross(static_cast<const Layer&>(lhs), static_cast<const Layer&>(rhs), engine, mutationRate);
    return std::unique_ptr<Model>(static_cast<Model*>(result.release()));
  }

}

