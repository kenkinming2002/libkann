#include <iterator>
#include <libkann/Model.hpp>

namespace kann
{
  Model::Model(const Model& other)
    : m_layers(other.m_layers)
  {
    for(auto& layer : m_layers)
      layer = layer->clone();
  }

  std::vector<std::shared_ptr<const Parameter>> Model::parameters(unsigned tags) const
  {
    std::vector<std::shared_ptr<const Parameter>> result;
    for(const auto& layer : m_layers)
      if(layer->tag() & tags)
      {
        auto parameters = layer->parameters(TAG_ALL);
        result.insert(result.end(),
          std::move_iterator(parameters.begin()),
          std::move_iterator(parameters.end())
        );
      }

    return result;
  }

  std::vector<std::shared_ptr<Parameter>> Model::parameters(unsigned tags)
  {
    std::vector<std::shared_ptr<Parameter>> result;
    for(const auto& layer : m_layers)
      if(layer->tag() & tags)
      {
        auto parameters = layer->parameters(TAG_ALL);
        result.insert(result.end(),
          std::move_iterator(parameters.begin()),
          std::move_iterator(parameters.end())
        );
      }

    return result;
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

