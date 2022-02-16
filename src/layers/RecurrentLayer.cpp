#include <libkann/layers/RecurrentLayer.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <assert.h>

namespace kann
{
  RecurrentLayer::RecurrentLayer(size_t memory)
    : m_memory(memory) {}

  void RecurrentLayer::addLayer(std::shared_ptr<const Layer> layer, Tag tag)
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

  std::vector<QualifiedName> RecurrentLayer::parameters(Scope scope) const
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

  std::vector<QualifiedName> RecurrentLayer::states(Scope scope) const
  {
    std::vector<QualifiedName> result;

    size_t i = 0;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto layerScope = this->layerScope(i++);
      auto layerStates = layer->states(scope + layerScope);
      result.insert(result.end(), layerStates.begin(), layerStates.end());
    }

    auto memoryParameter = QualifiedName{
      .name = "memory",
      .size = m_memory
    };
    result.push_back(memoryParameter);

    return result;
  }

  Layer::Output RecurrentLayer::process(Scope scope, Input input) const
  {
    auto memoryParameter = QualifiedName{
      .scope = scope,
      .name = "memory",
      .size = m_memory
    };

    auto [inputVariable, parameterVariables, inputStateVariables] = std::move(input);

    // 1: Concat input
    {
      const size_t realInputSize = m_taggedLayers.front().layer->inputSize();

      auto memoryVariable = inputStateVariables.at(memoryParameter);

      auto expandedInputVariable = std::make_shared<const Variable>(
        std::vector{inputVariable},
        std::make_shared<IdentityOperation>(realInputSize-m_memory, realInputSize, 0)
      );

      auto expandedMemoryVariable = std::make_shared<const Variable>(
        std::vector{memoryVariable},
        std::make_shared<IdentityOperation>(m_memory, realInputSize, realInputSize-m_memory)
      );

      auto realInputVariable = std::make_shared<const Variable>(
        std::vector{expandedInputVariable, expandedMemoryVariable},
        std::make_shared<ReduceOperation>(2)
      );

      inputVariable = std::move(realInputVariable);
    }

    auto outputVariable       = inputVariable;
    auto outputStateVariables = inputStateVariables;

    // 2: Pass through layers
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

    // 3: Split output
    {
      const size_t realOutputSize = m_taggedLayers.back().layer->outputSize();

      auto realOutputVariable = std::move(outputVariable);
      outputVariable = std::make_shared<const Variable>(
        std::vector{realOutputVariable},
        std::make_shared<IdentityOperation>(realOutputSize, realOutputSize - m_memory, 0)
      );

      outputStateVariables.at(memoryParameter) = std::make_shared<const Variable>(
        std::vector{realOutputVariable},
        std::make_shared<IdentityOperation>(realOutputSize, m_memory, realOutputSize - m_memory)
      );
    }

    return Output{
      std::move(outputVariable),
      std::move(outputStateVariables)
    };
  }
}

