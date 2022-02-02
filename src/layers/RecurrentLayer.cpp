#include <libkann/layers/RecurrentLayer.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <assert.h>

namespace kann
{
  RecurrentLayer::RecurrentLayer(size_t memory)
    : m_memory(memory) {}

  void RecurrentLayer::addLayer(std::shared_ptr<const NewLayer> layer)
  {
    m_layers.push_back(std::move(layer));
  }

  size_t RecurrentLayer::inputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.front()->inputSize();
  }

  size_t RecurrentLayer::outputSize() const
  {
    assert(!m_layers.empty());
    return m_layers.back()->outputSize();
  }

  std::vector<NewParameter> RecurrentLayer::parameters() const
  {
    std::vector<NewParameter> result;

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

  std::vector<NewParameter> RecurrentLayer::stateParameters() const
  {
    std::vector<NewParameter> result;

    size_t i = 0;
    for(const auto& layer : m_layers)
    {
      auto layerScope = this->layerScope(i++);
      auto layerStateParameters = layer->stateParameters();
      for(const auto& parameter : layerStateParameters)
        result.push_back(parameter.inScope(layerScope));
    }

    auto memoryParameter = NewParameter{
      .name = "memory",
      .size = m_memory
    };
    result.push_back(memoryParameter);

    return result;
  }

  LayerVariable RecurrentLayer::operator()(Scope scope, LayerVariable input) const
  {
    auto memoryParameter = NewParameter{
      .scope = scope,
      .name = "memory",
      .size = m_memory
    };

    // 1: Concat input
    {
      const size_t realInputSize = m_layers.front()->inputSize();

      auto inputVariable = std::move(input.variable);
      auto memoryVariable = input.lookup(LayerVariable::Type::STATE, memoryParameter);

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

      input.variable = std::move(realInputVariable);
    }


    // 2: Pass through layers
    auto output = std::move(input);

    size_t i = 0;
    for(const auto& layer : m_layers)
    {
      auto layerScope = this->layerScope(i++);
      output = (*layer)(scope+layerScope, output);
    }

    // 3: Split output
    {
      const size_t realOutputSize = m_layers.back()->outputSize();

      auto realOutputVariable = std::move(output.variable);
      auto outputVariable = std::make_shared<const Variable>(
        std::vector{realOutputVariable},
        std::make_shared<IdentityOperation>(realOutputSize, realOutputSize - m_memory, 0)
      );

      auto memoryVariable = std::make_shared<const Variable>(
        std::vector{realOutputVariable},
        std::make_shared<IdentityOperation>(realOutputSize, m_memory, realOutputSize - m_memory)
      );

      output.variable = std::move(outputVariable);
      output.assign(LayerVariable::Type::STATE, memoryParameter, std::move(memoryVariable));
    }

    return output;
  }
}

