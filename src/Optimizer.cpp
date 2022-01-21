#include <functional>
#include <libkann/Optimizer.hpp>

#include <libkann/Differentiate.hpp>
#include <libkann/DefaultExecutor.hpp>

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

  Optimizer::Optimizer(std::shared_ptr<Model> model, double learningRate)
    : m_model(std::move(model)), m_learningRate(learningRate)
  {
    auto parameters = m_model->parameters(TAG_ALL);
    auto states     = m_model->makeStates();

    // 1: Create Executor
    m_executor = makeDefaultExecutor();

    auto inputVariable = std::make_shared<const Variable>();
    m_executor->addInput("input", {inputVariable});

    auto expectedOutputVariable = std::make_shared<const Variable>();
    m_executor->addInput("expected output", {expectedOutputVariable});

    auto parametersVariables = convert(parameters, [](const std::shared_ptr<Parameter>& parameter){ return parameter->variable; });
    m_executor->addInput("parameters", parametersVariables);

    auto statesVariables = convert(states, [](const std::shared_ptr<Parameter>& parameter){ return parameter->variable; });
    m_executor->addInput("states", statesVariables);

    auto [outputVariable, newStatesVariables] = (*m_model)(inputVariable, statesVariables);
    m_executor->addOutput("output", {outputVariable});
    m_executor->addOutput("new states", newStatesVariables);

    auto outputGradientVariable = std::make_shared<const Variable>(
      std::vector{std::make_shared<const Variable>(
        std::vector{outputVariable,expectedOutputVariable},
        std::make_shared<SubtractOperation>()
      )},
      std::make_shared<MultiplyOperation>(2.0)
    );

    std::vector<std::shared_ptr<const Variable>> parametersGradientVariables;
    {
      auto gradientVariablesMap = differentiate({outputVariable}, {outputGradientVariable});
      for(auto& v : parametersVariables)
        parametersGradientVariables.push_back(gradientVariablesMap.at(v));
    }

    std::vector<std::shared_ptr<const Variable>> newParametersVariables;
    for(size_t i=0; i<parametersVariables.size(); ++i)
    {
      auto tmp = std::make_shared<const Variable>(std::vector{parametersGradientVariables[i]}, std::make_shared<MultiplyOperation>(learningRate));
      auto result = std::make_shared<const Variable>(std::vector{parametersVariables[i], std::move(tmp)}, std::make_shared<SubtractOperation>());
      newParametersVariables.push_back(std::move(result));
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

  std::pair<std::shared_ptr<const Tensor>, double> Optimizer::optimize(std::shared_ptr<const Tensor> input, std::shared_ptr<const Tensor> expectedOutput, unsigned tag)
  {
    auto parameters          = m_model->parameters(TAG_ALL);
    auto trainableParameters = m_model->parameters(tag);

    // 1: Input
    m_executor->input("input", {input});
    m_executor->input("expected output", {expectedOutput});
    m_executor->input("states", m_statesValues);

    auto parametersValues = convert(parameters, [](const std::shared_ptr<Parameter>& parameter){ return parameter->value; });
    m_executor->input("parameters", std::move(parametersValues));

    // 2: Output
    auto output = m_executor->output("output").front();

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

    // 5: Return
    // TODO: May be also put that into the computational graph
    const double cost = ((output->asVector()-expectedOutput->asVector()) * 2.0).squaredNorm();
    return std::make_pair(std::move(output), cost);
  }
}
