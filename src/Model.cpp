#include <libkann/Model.hpp>

#include <libkann/Differentiate.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerVariable.hpp>

#include <libkann/operations/SubtractOperation.hpp>
#include <libkann/operations/MultiplyOperation.hpp>

#include <fstream>

namespace kann
{
  Model::Model(std::shared_ptr<const Layer> layer)
    : m_layer(std::move(layer))
  {
    // Parameters
    auto parameters = m_layer->parameters(Scope());
    for(const auto& parameter : parameters)
    {
      auto value = std::make_shared<Tensor>(parameter.size);
      value->asArray().setZero();
      m_parametersMap.emplace(parameter, std::move(value));
    }

    // State Parameters
    auto stateParameters = m_layer->stateParameters(Scope());
    for(const auto& parameter : stateParameters)
    {
      auto value = std::make_shared<Tensor>(parameter.size);
      value->asArray().setZero();
      m_statesMap.emplace(parameter, std::move(value));
    }
  }

  void Model::randomize()
  {
    for(auto& [parameter, value] : m_parametersMap)
      value->asArray().setRandom();

    for(auto& [parameter, value] : m_statesMap)
      value->asArray().setRandom();
  }

  std::shared_ptr<const Tensor> Model::predict(std::shared_ptr<const Tensor> input)
  {
    if(!m_predictExecutor)
    {
      // 1: Pass variables through layer
      LayerVariable inputLayerVariable;
      inputLayerVariable.variable = std::make_shared<const Variable>();

      for(const auto& [parameter, value] : m_parametersMap)
        inputLayerVariable.insert(LayerVariable::Type::PARAMETER, parameter);

      for(const auto& [parameter, value] : m_statesMap)
        inputLayerVariable.insert(LayerVariable::Type::STATE, parameter);

      LayerVariable outputLayerVariable = (*m_layer)(Scope(), inputLayerVariable);

      // 2: Retrieve variables
      auto inputVariable  = inputLayerVariable.variable;
      auto outputVariable = outputLayerVariable.variable;

      m_predictExecutor = makeDefaultExecutor();

      m_predictExecutor->addInput("input",   {inputVariable});
      m_predictExecutor->addOutput("output", {outputVariable});

      // It should not matter whether we look it up from inputLayerVariable or
      // outputLayerVariable
      for(auto& [parameter, variable] : inputLayerVariable.parameterVariables)
        m_predictExecutor->addInput("parameters_input:"+parameter.qualifiedName(), {variable});

      for(auto& [parameter, variable] : inputLayerVariable.stateVariables)
        m_predictExecutor->addInput("states_input:"+parameter.qualifiedName(), {variable});

      for(auto& [parameter, variable] : outputLayerVariable.stateVariables)
        m_predictExecutor->addOutput("states_output:"+parameter.qualifiedName(), {variable});

      m_predictExecutor->build();
    }

    for(auto& [parameter, value] : m_parametersMap)
      m_predictExecutor->input("parameters_input:"+parameter.qualifiedName(), {value});

    for(auto& [parameter, value] : m_statesMap)
      m_predictExecutor->input("states_input:"+parameter.qualifiedName(), {value});

    m_predictExecutor->input("input", {input});

    for(auto& [parameter, value] : m_statesMap)
      value = m_predictExecutor->output("states_output:"+parameter.qualifiedName()).front();

    return m_predictExecutor->output("output").front();
  }

  std::pair<std::vector<std::shared_ptr<const Tensor>>, std::vector<double>> Model::optimize(
    double learningRate, Tag tag,
    std::vector<std::shared_ptr<const Tensor>> inputs,
    std::vector<std::shared_ptr<const Tensor>> expectedOutputs)
  {
    assert(inputs.size() == expectedOutputs.size());
    size_t batchSize = inputs.size();
    auto& optimizeExecutor = this->optimizeExecutor(learningRate, tag, batchSize);

    for(auto& [parameter, value] : m_parametersMap)
      optimizeExecutor.input("parameters_input:"+parameter.qualifiedName(), {value});

    for(auto& [parameter, value] : m_statesMap)
      optimizeExecutor.input("states_input:"+parameter.qualifiedName(), {value});

    optimizeExecutor.input("inputs", inputs);
    optimizeExecutor.input("expected outputs", expectedOutputs);

    for(auto& [parameter, value] : m_parametersMap)
      if(static_cast<bool>(parameter.scope.tag() & tag))
        value = optimizeExecutor.output("parameters_output:"+parameter.qualifiedName()).front();

    for(auto& [parameter, value] : m_statesMap)
      value = optimizeExecutor.output("states_output:"+parameter.qualifiedName()).front();

    auto outputs = optimizeExecutor.output("outputs");

    // Compute cost with outputs and expected outputs
    std::vector<double> costs;
    costs.reserve(outputs.size());

    for(size_t i=0; i<outputs.size(); ++i)
      costs.push_back(((outputs[i]->asVector() - expectedOutputs[i]->asVector()) * 2.0).squaredNorm());

    return {outputs, costs};
  }

  Executor& Model::optimizeExecutor(double learningRate, Tag tag, size_t batchSize)
  {
    auto config = OptimizeConfig{
      .learningRate = learningRate,
      .tag          = tag,
      .batchSize    = batchSize
    };

    if(auto it = m_optimizeExecutors.find(config); it != m_optimizeExecutors.end())
      return *it->second;

    // 1: Parameters and states variables
    std::unordered_map<Parameter, std::shared_ptr<const Variable>> inputParameterVariablesMap;
    std::unordered_map<Parameter, std::shared_ptr<const Variable>> outputParameterVariablesMap;

    std::unordered_map<Parameter, std::shared_ptr<const Variable>> inputStateVariablesMap;
    std::unordered_map<Parameter, std::shared_ptr<const Variable>> outputStateVariablesMap;

    for(const auto& [parameter, value] : m_parametersMap)
    {
      auto [it, success] = inputParameterVariablesMap.emplace(parameter, std::make_shared<const Variable>());
      assert(success);
    }

    for(const auto& [parameter, value] : m_statesMap)
    {
      auto [it, success] = inputStateVariablesMap.emplace(parameter, std::make_shared<const Variable>());
      assert(success);
    }
    outputStateVariablesMap = inputStateVariablesMap;

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
      inputLayerVariable.parameterVariables = inputParameterVariablesMap;
      inputLayerVariable.stateVariables     = std::move(outputStateVariablesMap);

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

      outputStateVariablesMap = std::move(outputLayerVariable.stateVariables);
    }

    // 3: New parameter variables
    auto gradientMap = differentiate(outputVariables, outputGradientVariables);
    for(const auto& [parameter, variable] : inputParameterVariablesMap)
    {
      auto gradient = gradientMap.at(variable);

      // FIXME: How should batchSize affect learningRate
      gradient = std::make_shared<const Variable>(std::vector{std::move(gradient)}, std::make_shared<MultiplyOperation>(learningRate * 0.5 * std::sqrt(batchSize)));
      auto newVariable = std::make_shared<const Variable>(std::vector{variable, std::move(gradient)}, std::make_shared<SubtractOperation>());
      auto [it, success] = outputParameterVariablesMap.emplace(parameter, std::move(newVariable));
      assert(success);
    }

    // 5: Create executor
    auto executor = makeDefaultExecutor();

    for(auto& [parameter, variable] : inputParameterVariablesMap)
      executor->addInput("parameters_input:"+parameter.qualifiedName(), {variable});

    for(auto& [parameter, variable] : outputParameterVariablesMap)
      executor->addOutput("parameters_output:"+parameter.qualifiedName(), {variable});

    for(auto& [parameter, variable] : inputStateVariablesMap)
      executor->addInput("states_input:"+parameter.qualifiedName(), {variable});

    for(auto& [parameter, variable] : outputStateVariablesMap)
      executor->addOutput("states_output:"+parameter.qualifiedName(), {variable});

    executor->addInput("inputs"          , std::move(inputVariables));
    executor->addInput("expected outputs", std::move(expectedOutputVariables));
    executor->addOutput("outputs"        , std::move(outputVariables));

    executor->build();
    {
      std::ofstream testFile("output/test.dot");
      executor->write_graphviz(testFile);
    }

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
    for(auto& [parameter, value] : result->m_parametersMap)
      value = cross(
        lhs.m_parametersMap.at(parameter),
        rhs.m_parametersMap.at(parameter),
        engine, mutationRate
      );

    return result;
  }
}
