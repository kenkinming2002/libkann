#include <libkann/NewModel.hpp>

#include <libkann/NewLayer.hpp>
#include <libkann/LayerVariable.hpp>

namespace kann
{
  NewModel::NewModel(std::shared_ptr<const NewLayer> layer)
    : m_layer(std::move(layer))
  {
    // Parameters
    auto parameters = m_layer->parameters();
    for(const auto& parameter : parameters)
      m_parameters.push_back(std::make_shared<const Tensor>(parameter.size));

    for(auto& value : m_parameters)
      value->asArray().setZero();

    // State Parameters
    auto stateParameters = m_layer->stateParameters();
    for(const auto& parameter : stateParameters)
      m_states.push_back(std::make_shared<const Tensor>(parameter.size));

    for(auto& value : m_states)
      value->asArray().setZero();
  }

  std::shared_ptr<const Tensor> NewModel::predict(std::shared_ptr<const Tensor> input)
  {
    if(!m_predictExecutor)
    {
      auto parameters      = m_layer->parameters();
      auto stateParameters = m_layer->stateParameters();

      // 1: Pass variables through layer
      LayerVariable inputLayerVariable;
      inputLayerVariable.variable = std::make_shared<const Variable>();

      for(auto& parameter : parameters)
        inputLayerVariable.insert(LayerVariable::Type::PARAMETER, std::move(parameter));

      for(auto& parameter : stateParameters)
        inputLayerVariable.insert(LayerVariable::Type::STATE, std::move(parameter));

      LayerVariable outputLayerVariable = (*m_layer)(Scope(), std::move(inputLayerVariable));

      // 2: Retrieve variables
      auto inputVariable  = inputLayerVariable.variable;
      auto outputVariable = outputLayerVariable.variable;

      std::vector<std::shared_ptr<const Variable>> parameterVariables;
      for(auto& parameter : parameters)
      {
        // It should not matter whether we look it up from inputLayerVariable or
        // outputLayerVariable
        auto variable = inputLayerVariable.lookup(LayerVariable::Type::PARAMETER, parameter);
        parameterVariables.push_back(std::move(variable));
      }

      std::vector<std::shared_ptr<const Variable>> stateVariables;
      for(auto& parameter : stateParameters)
      {
        // Lookup old state from inputLayerVariable
        auto variable = inputLayerVariable.lookup(LayerVariable::Type::STATE, parameter);
        stateVariables.push_back(std::move(variable));
      }

      std::vector<std::shared_ptr<const Variable>> newStateVariables;
      for(auto& parameter : stateParameters)
      {
        // Lookup new state from outputLayerVariable
        auto variable = outputLayerVariable.lookup(LayerVariable::Type::STATE, parameter);
        stateVariables.push_back(std::move(variable));
      }

      m_predictExecutor = makeDefaultExecutor();

      m_predictExecutor->addInput("parameters",  std::move(parameterVariables));
      m_predictExecutor->addInput("states",      std::move(stateVariables));
      m_predictExecutor->addOutput("new states", std::move(newStateVariables));

      m_predictExecutor->addInput("input",   {inputVariable});
      m_predictExecutor->addOutput("output", {outputVariable});
    }

    m_predictExecutor->input("parameters", m_parameters);
    m_predictExecutor->input("states",     m_states);
    m_predictExecutor->input("input", {input});

    auto output    = m_predictExecutor->output("output").front();
    auto newStates = m_predictExecutor->output("new states");

    m_states = std::move(newStates);
    return output;
  }
}
