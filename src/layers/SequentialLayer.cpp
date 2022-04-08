#include <libkann/layers/SequentialLayer.hpp>

#include <assert.h>

namespace kann
{
  void SequentialLayer::addLayer(CRef<Layer> layer, Tag tag)
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

  std::vector<Layer::Parameter> SequentialLayer::parameters() const
  {
    std::vector<Parameter> result;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto layerParameters = layer->parameters();
      result.insert(result.end(), layerParameters.begin(), layerParameters.end());
    }
    return result;
  }

  std::vector<Layer::State> SequentialLayer::states() const
  {
    std::vector<State> result;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto layerStates = layer->states();
      result.insert(result.end(), layerStates.begin(), layerStates.end());
    }
    return result;
  }

  Layer::ProcessOutput SequentialLayer::process(ProcessInput input) const
  {
    ProcessOutput output;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      output = layer->process(input);
      input.variable = output.variable;
      input.states   = output.states;
    }
    return output;
  }
}
