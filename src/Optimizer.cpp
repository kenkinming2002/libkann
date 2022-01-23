#include <functional>
#include <libkann/Optimizer.hpp>

#include <libkann/Differentiate.hpp>
#include <libkann/executors/DefaultExecutor.hpp>

#include <libkann/operations/SubtractOperation.hpp>
#include <libkann/operations/MultiplyOperation.hpp>

#include <iterator>
#include <fstream>

namespace kann
{
  template<typename U, typename UnaryFunc, typename T = std::result_of_t<UnaryFunc(const U&)>>
  static std::vector<T> convert(const std::vector<U>& in, const UnaryFunc& f)
  {
    std::vector<T> out;
    out.reserve(in.size());
    for(const auto& v : in)
      out.push_back(f(v));

    return out;
  }

  Optimizer::Optimizer(std::shared_ptr<Model> model, double learningRate, size_t batchSize)
    : m_model(std::move(model)), m_learningRate(learningRate), m_batchSize(batchSize)
  {
    auto parameters = m_model->parameters(TAG_ALL);
    auto states     = m_model->makeStates();

    // 1: Create Executor
    m_executor = makeDefaultExecutor();

    // Internal states and paramteres
    auto parametersVariables = convert(parameters, [](const std::shared_ptr<Parameter>& parameter){ return parameter->variable; });
    m_executor->addInput("parameters", parametersVariables);

    auto statesVariables = convert(states, [](const std::shared_ptr<Parameter>& parameter){ return parameter->variable; });
    m_executor->addInput("states", statesVariables);

    // There are multiple input and output variables
    std::vector<std::shared_ptr<const Variable>> inputVariables;
    std::vector<std::shared_ptr<const Variable>> expectedOutputVariables;
    std::vector<std::shared_ptr<const Variable>> outputVariables;
    std::vector<std::shared_ptr<const Variable>> outputGradientVariables;

    for(size_t i=0; i<batchSize; ++i)
    {
      auto inputVariable = std::make_shared<const Variable>();
      auto expectedOutputVariable = std::make_shared<const Variable>();
      auto [outputVariable, newStatesVariables] = (*m_model)(inputVariable, statesVariables);

      auto outputGradientVariable = std::make_shared<const Variable>(
        std::vector{std::make_shared<const Variable>(
          std::vector{outputVariable,expectedOutputVariable},
          std::make_shared<SubtractOperation>()
        )},
        std::make_shared<MultiplyOperation>(2.0)
      );

      inputVariables.push_back(std::move(inputVariable));
      outputVariables.push_back(std::move(outputVariable));
      expectedOutputVariables.push_back(std::move(expectedOutputVariable));
      outputGradientVariables.push_back(std::move(outputGradientVariable));

      statesVariables = std::move(newStatesVariables);
    }

    m_executor->addInput("inputs",  inputVariables);
    m_executor->addInput("expected outputs", expectedOutputVariables);
    m_executor->addOutput("outputs", outputVariables);
    m_executor->addOutput("new states", statesVariables);

    auto gradientMap = differentiate(outputVariables, outputGradientVariables);
    std::vector<std::shared_ptr<const Variable>> newParametersVariables;
    for(const auto& variable : parametersVariables)
    {
      auto gradient = gradientMap.at(variable);

      // FIXME: How should batchSize affect learningRate
      gradient = std::make_shared<const Variable>(std::vector{std::move(gradient)}, std::make_shared<MultiplyOperation>(m_learningRate * 0.5 * std::sqrt(m_batchSize)));
      auto newVariable = std::make_shared<const Variable>(std::vector{variable, std::move(gradient)}, std::make_shared<SubtractOperation>());
      newParametersVariables.push_back(std::move(newVariable));
    }

    m_executor->addOutput("new parameters", newParametersVariables);

    m_executor->build();

    // 2: Create initial state
    auto statesValues = convert(states, [](const std::shared_ptr<Parameter>& parameter){ return parameter->value; });
    m_statesValues = std::move(statesValues);

    // 3: DEBUG
    std::ofstream file("output/optimizer.dot");
    m_executor->write_graphviz(file);
  }

  std::vector<std::pair<std::shared_ptr<const Tensor>, double>> Optimizer::optimize(
    std::vector<std::shared_ptr<const Tensor>> inputs,
    std::vector<std::shared_ptr<const Tensor>> expectedOutputs,
    unsigned tag)
  {
    auto parameters          = m_model->parameters(TAG_ALL);
    auto trainableParameters = m_model->parameters(tag);

    // 1: Input
    m_executor->input("inputs", inputs);
    m_executor->input("expected outputs", expectedOutputs);
    m_executor->input("states", m_statesValues);

    auto parametersValues = convert(parameters, [](const std::shared_ptr<Parameter>& parameter){ return parameter->value; });
    m_executor->input("parameters", std::move(parametersValues));

    // 2: Output
    auto outputs = m_executor->output("outputs");

    // Parameters
    auto newParametersValues = m_executor->output("new parameters");

    std::unordered_set<std::shared_ptr<Parameter>> trainableParametersSet;
    for(auto& trainableParameter : trainableParameters)
      trainableParametersSet.insert(trainableParameter);

    // Only update trainable parameters
    for(size_t i=0; i<parameters.size(); ++i)
      if(trainableParametersSet.contains(parameters[i]))
        parameters[i]->value = newParametersValues[i];

    // States
    auto newStatesValues = m_executor->output("new states");
    m_statesValues = std::move(newStatesValues);

    // 3: Cost
    // TODO: May be also put that into the computational graph
    std::vector<double> costs;
    {
      assert(outputs.size() == expectedOutputs.size());
      const size_t size = outputs.size();
      costs.reserve(size);
      for(size_t i=0; i<size; ++i)
        costs.push_back((2.0 * (outputs[i]->asVector() - expectedOutputs[i]->asVector())).squaredNorm());
    }

    std::vector<std::pair<std::shared_ptr<const Tensor>, double>> result;
    {
      assert(outputs.size() == costs.size());
      const size_t size = outputs.size();
      result.reserve(size);
      for(size_t i=0; i<size; ++i)
        result.emplace_back(std::move(outputs[i]), costs[i]);
    }

    return result;
  }
}
