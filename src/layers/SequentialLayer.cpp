#include <libkann/layers/SequentialLayer.hpp>

#include <assert.h>

namespace kann
{
  void SequentialLayer::addLayer(std::shared_ptr<const Layer> layer)
  {
    m_layers.push_back(std::move(layer));
  }

  size_t SequentialLayer::inputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.front()->inputSize();
  }

  size_t SequentialLayer::outputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.back()->outputSize();
  }

  std::vector<Parameter> SequentialLayer::parameters() const
  {
    std::vector<Parameter> result;

    size_t i = 0;
    for(const auto& layer : m_layers)
    {
      auto layerScope = this->layerScope(i++);
      auto layerParameters = layer->parameters();
      for(const auto& parameter : layerParameters)
        result.push_back(parameter.inScope(layerScope));
    }
    return result;
  }

  std::vector<Parameter> SequentialLayer::stateParameters() const
  {
    std::vector<Parameter> result;

    size_t i = 0;
    for(const auto& layer : m_layers)
    {
      auto layerScope = this->layerScope(i++);
      auto layerStateParameters = layer->stateParameters();
      for(const auto& parameter : layerStateParameters)
        result.push_back(parameter.inScope(layerScope));
    }
    return result;
  }

  LayerVariable SequentialLayer::operator()(Scope scope, LayerVariable input) const
  {
    auto output = std::move(input);

    size_t i = 0;
    for(const auto& layer : m_layers)
    {
      auto layerScope = this->layerScope(i++);
      output = (*layer)(scope+layerScope, output);
    }

    return output;
  }
}
