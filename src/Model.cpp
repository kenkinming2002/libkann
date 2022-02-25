#include <libkann/Model.hpp>

#include <libkann/Differentiate.hpp>

#include <libkann/Layer.hpp>

#include <libkann/operations/SubtractOperation.hpp>
#include <libkann/operations/MultiplyOperation.hpp>

#include <fstream>
#include <random>

namespace kann
{
  Model::Model(std::shared_ptr<const Layer> layer)
    : m_layer(std::move(layer))
  {
    // Parameters
    auto parameters = m_layer->parameters(Scope());
    for(auto& parameter : parameters)
    {
      ParameterInfo info;

      info.value = std::make_shared<Tensor>(parameter.size);
      info.value->asArray().setZero();
      info.size = parameter.size;
      info.mean = parameter.mean;
      info.stddev = parameter.stddev;

      m_parametersMap.emplace(std::move(parameter.name), std::move(info));
    }

    // State Parameters
    auto states = m_layer->states(Scope());
    for(auto& state : states)
    {
      StateInfo info;

      info.value = std::make_shared<Tensor>(state.size);
      info.value->asArray().setZero();
      info.size = state.size;

      m_statesMap.emplace(std::move(state.name), std::move(info));
    }
  }

  void Model::randomize(std::default_random_engine& engine)
  {
    for(auto& [name, info] : m_parametersMap)
      info.value = [&, info=info]() {
        auto value = std::make_shared<Tensor>(info.size);

        std::normal_distribution dist(info.mean, info.stddev);
        value->asArray() = Eigen::ArrayXd::NullaryExpr(info.size, [&](){
          return dist(engine);
        });

        return value;
      }();

    for(auto& [name, info] : m_statesMap)
      info.value = [&, info=info]() {
        auto value = std::make_shared<Tensor>(info.size);
        value->asArray() = Eigen::ArrayXd::Zero(info.size);
        return value;
      }();
  }

  std::shared_ptr<const Tensor> Model::predict(std::shared_ptr<const Tensor> input)
  {
    if(!m_predictExecutor)
    {
      CRef<Variable> inputVariable = std::make_shared<const Variable>();

      Map<Variable> parameterVariables;
      for(const auto& [parameter, value] : m_parametersMap)
        parameterVariables.emplace(parameter, std::make_shared<const Variable>());

      Map<Variable> inputStateVariables;
      for(const auto& [parameter, value] : m_statesMap)
        inputStateVariables.emplace(parameter, std::make_shared<const Variable>());

      auto [outputVariable, outputStateVariables] = m_layer->process(Scope(), {
        inputVariable,
        parameterVariables,
        inputStateVariables
      });

      m_predictExecutor = Executor::create(Executor::Type::DEFAULT);

      m_predictExecutor->addInput("input",   {inputVariable});
      m_predictExecutor->addOutput("output", {outputVariable});

      // It should not matter whether we look it up from inputLayerVariable or
      // outputLayerVariable
      for(auto& [parameter, variable] : parameterVariables)
        m_predictExecutor->addInput("parameters_input:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : inputStateVariables)
        m_predictExecutor->addInput("states_input:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : outputStateVariables)
        m_predictExecutor->addOutput("states_output:"+parameter.toString(), {variable});

      m_predictExecutor->build();
    }

    for(auto& [name, info] : m_parametersMap)
      m_predictExecutor->input("parameters_input:"+name.toString(), {info.value});

    for(auto& [name, info] : m_statesMap)
      m_predictExecutor->input("states_input:"+name.toString(), {info.value});

    m_predictExecutor->input("input", {input});

    for(auto& [name, info] : m_statesMap)
      info.value = m_predictExecutor->output("states_output:"+name.toString()).front();

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

    for(auto& [name, info] : m_parametersMap)
      state.executor->input("parameters_input:"+name.toString(), {info.value});

    for(auto& [name, info] : m_statesMap)
      state.executor->input("states_input:"+name.toString(), {info.value});

    state.executor->input("inputs", inputs);
    state.executor->input("expected outputs", expectedOutputs);

    for(auto& [qualifiedName, value] : state.map)
      value = state.executor->output("optimizer_states_output:"+qualifiedName.toString()).front();

    for(auto& [name, info] : m_parametersMap)
      if(static_cast<bool>(name.scope.tag() & tag))
        info.value = state.executor->output("parameters_output:"+name.toString()).front();

    for(auto& [name, info] : m_statesMap)
      info.value = state.executor->output("states_output:"+name.toString()).front();

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
      Map<Variable> inputParameterVariables;
      Map<Variable> outputParameterVariables;

      Map<Variable> inputStateVariables;
      Map<Variable> outputStateVariables;

      for(const auto& [parameter, value] : m_parametersMap)
        inputParameterVariables.emplace(parameter, std::make_shared<const Variable>());

      for(const auto& [parameter, value] : m_statesMap)
        inputStateVariables.emplace(parameter, std::make_shared<const Variable>());

      outputStateVariables = inputStateVariables;


      // 2: Create input, expected output, output and output gradient variables by
      // passing them through layer
      std::vector<std::shared_ptr<const Variable>> inputVariables;
      std::vector<std::shared_ptr<const Variable>> expectedOutputVariables;
      std::vector<std::shared_ptr<const Variable>> outputVariables;
      std::vector<std::shared_ptr<const Variable>> outputGradientVariables;

      for(size_t i=0; i<batchSize; ++i)
      {
        // 1: Call Layer::process
        CRef<Variable> inputVariable = std::make_shared<const Variable>();
        auto [outputVariable, _outputStateVariables] = m_layer->process(Scope(), {
            inputVariable,
            inputParameterVariables,
            inputStateVariables
        });

        // 2: New state variables
        outputStateVariables = std::move(_outputStateVariables);

        // 3: Create expected output variable and compute gradient
        auto expectedOutputVariable = std::make_shared<const Variable>();
        auto outputGradientVariable = std::make_shared<const Variable>(
          std::vector{std::make_shared<const Variable>(
            std::vector{outputVariable,expectedOutputVariable},
            std::make_shared<SubtractOperation>()
          )},
          std::make_shared<MultiplyOperation>(2.0)
        );

        // 4: Retrive variables
        inputVariables.push_back(std::move(inputVariable));
        outputVariables.push_back(std::move(outputVariable));
        expectedOutputVariables.push_back(std::move(expectedOutputVariable));
        outputGradientVariables.push_back(std::move(outputGradientVariable));

      }

      // 3: New parameter variables
      Map<Variable> optimizerStateInputVariables;
      Map<Variable> optimizerStateOutputVariables;

      auto gradientMap = differentiate(outputVariables, outputGradientVariables);
      for(const auto& [qualifiedName, parameter] : inputParameterVariables)
      {
        auto& info = m_parametersMap.at(qualifiedName);

        Optimizer::Context context;
        context.parameter = Layer::Parameter{
          .name = qualifiedName,
          .size = info.size,
          .mean = info.mean,
          .stddev = info.stddev
        };
        context.gradient = gradientMap.at(parameter);
        context.inputParameter = parameter;

        optimizer->process(context);

        outputParameterVariables.emplace(qualifiedName, context.outputParameter);
        optimizerStateInputVariables.merge(context.inputState);
        optimizerStateOutputVariables.merge(context.outputState);

        optimizeState.map.merge(context.initialState);
      }

      // 5: Create executor
      auto executor = Executor::create(Executor::Type::THREADED);

      for(auto& [qualifiedName, variable] : optimizerStateInputVariables)
        executor->addInput("optimizer_states_input:"+qualifiedName.toString(), {variable});

      for(auto& [qualifiedName, variable] : optimizerStateOutputVariables)
        executor->addOutput("optimizer_states_output:"+qualifiedName.toString(), {variable});

      for(auto& [parameter, variable] : inputParameterVariables)
        executor->addInput("parameters_input:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : outputParameterVariables)
        if(static_cast<bool>(parameter.scope.tag() & tag))
          executor->addOutput("parameters_output:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : inputStateVariables)
        executor->addInput("states_input:"+parameter.toString(), {variable});

      for(auto& [parameter, variable] : outputStateVariables)
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

  std::shared_ptr<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate)
  {
    // They have to have the same underlying structure for cross to work
    assert(lhs.m_layer.get() == rhs.m_layer.get());

    auto result = std::make_shared<Model>(lhs.m_layer);
    for(auto& [name, info] : result->m_parametersMap)
    {
      const auto& lhsInfo = lhs.m_parametersMap.at(name);
      const auto& rhsInfo = rhs.m_parametersMap.at(name);

      auto newValue = std::make_shared<Tensor>(info.size);

      std::normal_distribution distWeight(info.mean, info.stddev);
      std::uniform_real_distribution distMutation(0.0,1.0);
      std::uniform_int_distribution distSelection(0,1);

      newValue->asArray() = lhsInfo.value->asArray().binaryExpr(rhsInfo.value->asArray(), [&](double a, double b){
        if(distMutation(engine)>=mutationRate)
          return distWeight(engine);
        else
          return distSelection(engine) == 0 ? a : b;
      });

      info.value = std::move(newValue);
    }

    return result;
  }
}
