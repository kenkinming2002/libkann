#include <libkann/layers/RecurrentLayer.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <assert.h>

namespace kann
{
  RecurrentLayer::RecurrentLayer(size_t memory)
    : m_memory(memory) {}

  void RecurrentLayer::addLayer(CRef<Layer> layer, Tag tag)
  {
    m_taggedLayers.push_back(TaggedLayer{
      .tag   = tag,
      .layer = std::move(layer)
    });
  }

  size_t RecurrentLayer::inputSize() const
  {
    assert(!m_taggedLayers.empty());
    return m_taggedLayers.front().layer->inputSize();
  }

  size_t RecurrentLayer::outputSize() const
  {
    assert(!m_taggedLayers.empty());
    return m_taggedLayers.back().layer->outputSize();
  }

  std::vector<Layer::Parameter> RecurrentLayer::parameters() const
  {
    std::vector<Parameter> result;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto layerParameters = layer->parameters();
      result.insert(result.end(), layerParameters.begin(), layerParameters.end());
    }
    return result;
  }

  std::vector<Layer::State> RecurrentLayer::states() const
  {
    std::vector<State> result;

    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto layerStates = layer->states();
      result.insert(result.end(), layerStates.begin(), layerStates.end());
    }

    auto memoryState = State{
      .layer = shared_from_this(),
      .name = "memory",
      .size = m_memory
    };
    result.push_back(memoryState);

    return result;
  }

  Layer::ProcessOutput RecurrentLayer::process(ProcessInput input) const
  {
    // 1: Concat input
    {
      const size_t realInputSize = m_taggedLayers.front().layer->inputSize();

      auto memoryVariable = input.states.at({shared_from_this(), "memory"});
      auto expandedInputVariable  = Variable::apply(IdentityOperation(realInputSize-m_memory, realInputSize, 0                     ), std::vector{input.variable});
      auto expandedMemoryVariable = Variable::apply(IdentityOperation(m_memory              , realInputSize, realInputSize-m_memory), std::vector{memoryVariable});
      input.variable = Variable::apply(ReduceOperation(2), std::vector{expandedInputVariable, expandedMemoryVariable});
    }

    ProcessOutput output;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      output = layer->process(input);
      input.variable = output.variable;
      input.states   = output.states;
    }

    // 3: Split output
    {
      const size_t realOutputSize = m_taggedLayers.back().layer->outputSize();
      auto outputVariable = Variable::apply(IdentityOperation(realOutputSize, realOutputSize - m_memory, 0                        ), std::vector{output.variable});
      auto memoryVariable = Variable::apply(IdentityOperation(realOutputSize, m_memory                 , realOutputSize - m_memory), std::vector{output.variable});
      output.variable = std::move(outputVariable);
      output.states.at({shared_from_this(), "memory"}) = std::move(memoryVariable);
    }

    return output;
  }
}

