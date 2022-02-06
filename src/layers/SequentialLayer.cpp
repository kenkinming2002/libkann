#include <libkann/layers/SequentialLayer.hpp>

#include <assert.h>

namespace kann
{
  void SequentialLayer::addLayer(std::shared_ptr<const Layer> layer, Tag tag)
  {
    m_taggedLayers.push_back(TaggedLayer{
      .layer = std::move(layer),
      .tag   = tag
    });
  }

  size_t SequentialLayer::inputSize() const
  {
    assert(!m_taggedLayers.empty());
    return m_taggedLayers.front().layer->inputSize();
  }

  size_t SequentialLayer::outputSize() const
  {
    assert(!m_taggedLayers.empty());
    return m_taggedLayers.back().layer->outputSize();
  }

  std::vector<QualifiedName> SequentialLayer::parameters(Scope scope) const
  {
    std::vector<QualifiedName> result;

    size_t i = 0;
    for(const auto& [layer, tag] : m_taggedLayers)
    {
      auto layerScope = this->layerScope(i++);
      auto layerParameters = layer->parameters(scope + layerScope);
      result.insert(result.end(), layerParameters.begin(), layerParameters.end());
    }
    return result;
  }

  std::vector<QualifiedName> SequentialLayer::states(Scope scope) const
  {
    std::vector<QualifiedName> result;

    size_t i = 0;
    for(const auto& [layer, tag] : m_taggedLayers)
    {
      auto layerScope = this->layerScope(i++);
      auto layerStates = layer->states(scope + layerScope);
      result.insert(result.end(), layerStates.begin(), layerStates.end());
    }
    return result;
  }

  LayerVariable SequentialLayer::operator()(Scope scope, LayerVariable input) const
  {
    auto output = std::move(input);

    size_t i = 0;
    for(const auto& [layer, tag] : m_taggedLayers)
    {
      auto layerScope = this->layerScope(i++);
      output = (*layer)(scope+layerScope, output);
    }

    return output;
  }
}
