#include <libkann/layers/SequentialLayer.hpp>

#include <assert.h>

namespace kann
{
  void SequentialLayer::addLayer(std::shared_ptr<const NewLayer> layer)
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

  std::vector<NewParameter> SequentialLayer::parameters() const
  {
    std::vector<NewParameter> result;

    size_t i = 0;
    for(const auto& layer : m_layers)
    {
      auto layerParameters = layer->parameters();
      for(auto& parameter : layerParameters)
        result.push_back(parameter.withPrefix("layer"+std::to_string(i)).withPrefix("sequential"));
    }
    return result;
  }

  std::vector<NewParameter> SequentialLayer::stateParameters() const
  {
    std::vector<NewParameter> result;

    size_t i = 0;
    for(const auto& layer : m_layers)
    {
      auto layerStateParameters = layer->stateParameters();
      for(auto& parameter : layerStateParameters)
        result.push_back(parameter.withPrefix("layer"+std::to_string(i)).withPrefix("sequential"));
    }
    return result;
  }

  LayerVariable SequentialLayer::operator()(LayerVariable input) const
  {
    auto output = std::move(input);
    for(const auto& layer : m_layers)
      output = (*layer)(std::move(output));

    return output;
  }
}
