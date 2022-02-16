#include <libkann/layers/SequentialLayer.hpp>

#include <assert.h>

namespace kann
{
  void SequentialLayer::addLayer(std::shared_ptr<const Layer> layer, Tag tag)
  {
    m_taggedLayers.push_back(TaggedLayer{
      .tag   = tag,
      .layer = std::move(layer)
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
    for(const auto& [tag, layer] : m_taggedLayers)
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
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto layerScope = this->layerScope(i++);
      auto layerStates = layer->states(scope + layerScope);
      result.insert(result.end(), layerStates.begin(), layerStates.end());
    }
    return result;
  }

  Layer::Output SequentialLayer::process(Scope scope, Input input) const
  {
    auto [inputVariable, parameterVariables, inputStateVariables] = std::move(input);
    auto outputVariable       = inputVariable;
    auto outputStateVariables = inputStateVariables;

    size_t i = 0;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto layerScope = this->layerScope(i++);
      auto [_outputVariable, _outputStateVariables] = layer->process(scope+layerScope, {
        std::move(outputVariable),
        parameterVariables,
        std::move(outputStateVariables)
      });
      outputVariable       = std::move(_outputVariable);
      outputStateVariables = std::move(_outputStateVariables);
    }

    return Output{
      std::move(outputVariable),
      std::move(outputStateVariables)
    };
  }
}
