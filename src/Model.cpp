#include <libkann/Model.hpp>

#include <libkann/Differentiate.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerVariable.hpp>

#include <libkann/operations/SubtractOperation.hpp>
#include <libkann/operations/MultiplyOperation.hpp>

namespace kann
{
  Model::Model(std::shared_ptr<const Layer> layer)
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

  void Model::randomize()
  {
    for(auto& value : m_parameters)
      value->asArray().setRandom();

    for(auto& value : m_states)
      value->asArray().setRandom();
  }

  std::shared_ptr<const Tensor> Model::predict(std::shared_ptr<const Tensor> input)
  {
    if(!m_predictExecutor)
    {
      auto parameters      = m_layer->parameters();
      auto stateParameters = m_layer->stateParameters();

      // 1: Pass variables through layer
      LayerVariable inputLayerVariable;
      inputLayerVariable.variable = std::make_shared<const Variable>();

      for(auto& parameter : parameters)
        inputLayerVariable.insert(LayerVariable::Type::PARAMETER, parameter);

      for(auto& parameter : stateParameters)
        inputLayerVariable.insert(LayerVariable::Type::STATE, parameter);

      LayerVariable outputLayerVariable = (*m_layer)(Scope(), inputLayerVariable);

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
        newStateVariables.push_back(std::move(variable));
      }

      m_predictExecutor = makeDefaultExecutor();

      m_predictExecutor->addInput("parameters",  std::move(parameterVariables));
      m_predictExecutor->addInput("states",      std::move(stateVariables));
      m_predictExecutor->addOutput("new states", std::move(newStateVariables));

      m_predictExecutor->addInput("input",   {inputVariable});
      m_predictExecutor->addOutput("output", {outputVariable});

      m_predictExecutor->build();
    }

    m_predictExecutor->input("parameters", m_parameters);
    m_predictExecutor->input("states",     m_states);
    m_predictExecutor->input("input", {input});

    auto output    = m_predictExecutor->output("output").front();
    auto newStates = m_predictExecutor->output("new states");

    m_states = std::move(newStates);
    return output;
  }

  std::pair<std::vector<std::shared_ptr<const Tensor>>, std::vector<double>> Model::optimize(
    double learningRate, unsigned tags,
    std::vector<std::shared_ptr<const Tensor>> inputs,
    std::vector<std::shared_ptr<const Tensor>> expectedOutputs)
  {
    assert(inputs.size() == expectedOutputs.size());
    size_t batchSize = inputs.size();
    auto& optimizeExecutor = this->optimizeExecutor(learningRate, tags, batchSize);


    optimizeExecutor.input("inputs", inputs);
    optimizeExecutor.input("expected outputs", expectedOutputs);
    optimizeExecutor.input("states", m_states);
    optimizeExecutor.input("parameters", m_parameters);

    m_states     = optimizeExecutor.output("new states");
    m_parameters = optimizeExecutor.output("new parameters");
    auto outputs = optimizeExecutor.output("outputs");

    // Compute cost with outputs and expected outputs
    std::vector<double> costs;
    costs.reserve(outputs.size());

    for(size_t i=0; i<outputs.size(); ++i)
      costs.push_back(((outputs[i]->asVector() - expectedOutputs[i]->asVector()) * 2.0).squaredNorm());

    return {outputs, costs};
  }

  Executor& Model::optimizeExecutor(double learningRate, unsigned tags, size_t batchSize)
  {
    auto config = OptimizeConfig{
      .learningRate = learningRate,
      .tags         = tags,
      .batchSize    = batchSize
    };

    if(auto it = m_optimizeExecutors.find(config); it != m_optimizeExecutors.end())
      return *it->second;

    // Create it
    auto parameters      = m_layer->parameters();
    auto stateParameters = m_layer->stateParameters();

    // 1: Parameters and states variables
    std::unordered_map<Parameter, std::shared_ptr<const Variable>> parameterVariablesMap;
    std::vector<std::shared_ptr<const Variable>> parameterVariables;
    for(const auto& parameter : parameters)
    {
      auto [it, success] = parameterVariablesMap.emplace(parameter, std::make_shared<const Variable>());
      assert(success);
      parameterVariables.push_back(it->second);
    }

    std::unordered_map<Parameter, std::shared_ptr<const Variable>> stateVariablesMap;
    std::vector<std::shared_ptr<const Variable>> stateVariables;
    for(const auto& parameter : stateParameters)
    {
      auto [it, success] = stateVariablesMap.emplace(parameter, std::make_shared<const Variable>());
      assert(success);
      stateVariables.push_back(it->second);
    }

    // 2: Create input, expected output, output and output gradient variables by
    // passing them through layer
    std::vector<std::shared_ptr<const Variable>> inputVariables;
    std::vector<std::shared_ptr<const Variable>> expectedOutputVariables;
    std::vector<std::shared_ptr<const Variable>> outputVariables;
    std::vector<std::shared_ptr<const Variable>> outputGradientVariables;

    for(size_t i=0; i<batchSize; ++i)
    {
      // 1: Pass variables through layer
      LayerVariable inputLayerVariable;

      inputLayerVariable.variable = std::make_shared<const Variable>();
      inputLayerVariable.parameterVariables = parameterVariablesMap;
      inputLayerVariable.stateVariables     = std::move(stateVariablesMap);

      LayerVariable outputLayerVariable = (*m_layer)(Scope(), inputLayerVariable);

      // 2: Create expected output variable and compute gradient
      auto inputVariable  = inputLayerVariable.variable;
      auto outputVariable = outputLayerVariable.variable;
      auto expectedOutputVariable = std::make_shared<const Variable>();
      auto outputGradientVariable = std::make_shared<const Variable>(
        std::vector{std::make_shared<const Variable>(
          std::vector{outputVariable,expectedOutputVariable},
          std::make_shared<SubtractOperation>()
        )},
        std::make_shared<MultiplyOperation>(2.0)
      );

      // 3: Retrive variables
      inputVariables.push_back(std::move(inputVariable));
      outputVariables.push_back(std::move(outputVariable));
      expectedOutputVariables.push_back(std::move(expectedOutputVariable));
      outputGradientVariables.push_back(std::move(outputGradientVariable));

      stateVariablesMap = std::move(outputLayerVariable.stateVariables);
    }

    // 3: New state variables
    std::vector<std::shared_ptr<const Variable>> newStateVariables;
    for(const auto& parameter : stateParameters)
    {
      auto it = stateVariablesMap.find(parameter);
      assert(it != stateVariablesMap.end());
      newStateVariables.push_back(it->second);
    }

    // 4: New parameter variables
    auto gradientMap = differentiate(outputVariables, outputGradientVariables);
    std::vector<std::shared_ptr<const Variable>> newParametersVariables;
    for(const auto& variable : parameterVariables)
    {
      auto gradient = gradientMap.at(variable);

      // FIXME: How should batchSize affect learningRate
      gradient = std::make_shared<const Variable>(std::vector{std::move(gradient)}, std::make_shared<MultiplyOperation>(learningRate * 0.5 * std::sqrt(batchSize)));
      auto newVariable = std::make_shared<const Variable>(std::vector{variable, std::move(gradient)}, std::make_shared<SubtractOperation>());
      newParametersVariables.push_back(std::move(newVariable));
    }

    // 5: Create executor
    auto executor = makeDefaultExecutor();
    executor->addInput("parameters", std::move(parameterVariables));
    executor->addInput("states"    , std::move(stateVariables));

    executor->addInput("inputs"          , std::move(inputVariables));
    executor->addInput("expected outputs", std::move(expectedOutputVariables));
    executor->addOutput("outputs"        , std::move(outputVariables));

    executor->addOutput("new states"    , std::move(newStateVariables));
    executor->addOutput("new parameters", std::move(newParametersVariables));

    executor->build();

    auto [it, success] = m_optimizeExecutors.emplace(config, std::move(executor));
    assert(success);
    return *it->second;
  }

  static std::shared_ptr<const Tensor> cross(const std::shared_ptr<const Tensor>& lhs, const std::shared_ptr<const Tensor>& rhs, std::default_random_engine& engine, double mutationRate)
  {
    assert(lhs->size() == rhs->size());

    // FIXME: Somehow determine the range
    std::uniform_real_distribution distWeight(-1.0, 1.0);
    std::uniform_real_distribution distMutation(0.0,1.0);
    std::uniform_int_distribution distSelection(0,1);

    auto result = std::make_shared<Tensor>(lhs->size());
    result->asArray() = lhs->asArray().binaryExpr(rhs->asArray(), [&](double a, double b){
      if(distMutation(engine)>=mutationRate)
        return distWeight(engine);
      else
        return distSelection(engine) == 0 ? a : b;
    });
    return result;
  }

  std::shared_ptr<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate)
  {
    // They have to have the same underlying structure for cross to work
    assert(lhs.m_layer.get() == rhs.m_layer.get());

    auto result = std::make_shared<Model>(lhs.m_layer);
    for(size_t i=0; i<result->m_parameters.size(); ++i)
      result->m_parameters[i] = cross(lhs.m_parameters[i], rhs.m_parameters[i], engine, mutationRate);

    return result;
  }
}
