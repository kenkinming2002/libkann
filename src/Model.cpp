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
    auto states = m_layer->states(Scope());
    for(const auto& parameter : states)
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
        m_predictExecutor->addInput("parameters_input:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : inputLayerVariable.stateVariables)
        m_predictExecutor->addInput("states_input:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : outputLayerVariable.stateVariables)
        m_predictExecutor->addOutput("states_output:"+parameter.toString(), {variable});

      m_predictExecutor->build();
    }

    for(auto& [parameter, value] : m_parametersMap)
      m_predictExecutor->input("parameters_input:"+parameter.toString(), {value});

    for(auto& [parameter, value] : m_statesMap)
      m_predictExecutor->input("states_input:"+parameter.toString(), {value});

    m_predictExecutor->input("input", {input});

    for(auto& [parameter, value] : m_statesMap)
      value = m_predictExecutor->output("states_output:"+parameter.toString()).front();

    return m_predictExecutor->output("output").front();
  }

  std::pair<std::vector<std::shared_ptr<const Tensor>>, std::vector<double>> Model::optimize(
    std::shared_ptr<const Optimizer> optimizer, Tag tag,
    std::vector<std::shared_ptr<const Tensor>> inputs,
    std::vector<std::shared_ptr<const Tensor>> expectedOutputs)
  {
    assert(inputs.size() == expectedOutputs.size());
    size_t batchSize = inputs.size();
    auto& state = this->optimizeState(optimizer, tag, batchSize);

    for(auto& [qualifiedName, value] : state.map)
      state.executor->input("optimizer_states_input:"+qualifiedName.toString(), {value});

    for(auto& [parameter, value] : m_parametersMap)
      state.executor->input("parameters_input:"+parameter.toString(), {value});

    for(auto& [parameter, value] : m_statesMap)
      state.executor->input("states_input:"+parameter.toString(), {value});

    state.executor->input("inputs", inputs);
    state.executor->input("expected outputs", expectedOutputs);

    for(auto& [qualifiedName, value] : state.map)
      value = state.executor->output("optimizer_states_output:"+qualifiedName.toString()).front();

    for(auto& [parameter, value] : m_parametersMap)
      if(static_cast<bool>(parameter.scope.tag() & tag))
        value = state.executor->output("parameters_output:"+parameter.toString()).front();

    for(auto& [parameter, value] : m_statesMap)
      value = state.executor->output("states_output:"+parameter.toString()).front();

    auto outputs = state.executor->output("outputs");

    // Compute cost with outputs and expected outputs
    std::vector<double> costs;
    costs.reserve(outputs.size());

    for(size_t i=0; i<outputs.size(); ++i)
      costs.push_back(((outputs[i]->asVector() - expectedOutputs[i]->asVector()) * 2.0).squaredNorm());

    return {outputs, costs};
  }

  Model::OptimizeState& Model::optimizeState(std::shared_ptr<const Optimizer> optimizer, Tag tag, size_t batchSize)
  {
    auto config = OptimizeConfig{
      .optimizer = optimizer,
      .tag       = tag,
      .batchSize = batchSize
    };

    OptimizeState& optimizeState = m_optimizeStates[config];
    if(!optimizeState.executor)
    {
      // 1: Parameters and states variables
      std::unordered_map<QualifiedName, std::shared_ptr<const Variable>> inputParameterVariablesMap;
      std::unordered_map<QualifiedName, std::shared_ptr<const Variable>> outputParameterVariablesMap;

      std::unordered_map<QualifiedName, std::shared_ptr<const Variable>> inputStateVariablesMap;
      std::unordered_map<QualifiedName, std::shared_ptr<const Variable>> outputStateVariablesMap;

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
      VMap optimizerStateInputVariables;
      VMap optimizerStateOutputVariables;

      auto gradientMap = differentiate(outputVariables, outputGradientVariables);
      for(const auto& [qualifiedName, parameter] : inputParameterVariablesMap)
      {
        Optimizer::Context context;
        context.qualifiedName = qualifiedName;
        context.gradient = gradientMap.at(parameter);
        context.inputParameter = parameter;

        optimizer->process(context);

        outputParameterVariablesMap.emplace(qualifiedName, context.outputParameter);
        optimizerStateInputVariables.merge(context.inputState);
        optimizerStateOutputVariables.merge(context.outputState);

        optimizeState.map.merge(context.initialState);
      }

      // 5: Create executor
      auto executor = makeDefaultExecutor();

      for(auto& [qualifiedName, variable] : optimizerStateInputVariables)
        executor->addInput("optimizer_states_input:"+qualifiedName.toString(), {variable});

      for(auto& [qualifiedName, variable] : optimizerStateOutputVariables)
        executor->addOutput("optimizer_states_output:"+qualifiedName.toString(), {variable});

      for(auto& [parameter, variable] : inputParameterVariablesMap)
        executor->addInput("parameters_input:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : outputParameterVariablesMap)
        if(static_cast<bool>(parameter.scope.tag() & tag))
          executor->addOutput("parameters_output:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : inputStateVariablesMap)
        executor->addInput("states_input:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : outputStateVariablesMap)
        executor->addOutput("states_output:"+parameter.toString(), {variable});

      executor->addInput("inputs"          , std::move(inputVariables));
      executor->addInput("expected outputs", std::move(expectedOutputVariables));
      executor->addOutput("outputs"        , std::move(outputVariables));

      executor->build();
      {
        std::ofstream testFile("output/test.dot");
        executor->write_graphviz(testFile);
      }

      optimizeState.executor = std::move(executor);
    }

    return optimizeState;
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
