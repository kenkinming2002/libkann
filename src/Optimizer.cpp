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
  Optimizer::Optimizer(std::shared_ptr<Model> model, double learningRate)
    : m_model(std::move(model)), m_learningRate(learningRate)
  {
    // 1: Create Executor
    m_executor = makeDefaultExecutor();

    auto inputVariable = std::make_shared<const Variable>();
    m_executor->addInput("input", {inputVariable});

    auto expectedOutputVariable = std::make_shared<const Variable>();
    m_executor->addInput("expected output", {expectedOutputVariable});

    auto parametersVariables = m_model->parametersVariables(TAG_ALL);
    m_executor->addInput("parameters", parametersVariables);

    auto stateVariables = m_model->makeStateVariables();
    m_executor->addInput("state", stateVariables);

    auto [outputVariable, newStateVariables] = (*m_model)(inputVariable, stateVariables);
    m_executor->addOutput("output", {outputVariable});
    m_executor->addOutput("new state", newStateVariables);

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
    m_state = m_model->makeState();

    // 3: DEBUG
    std::ofstream file("output/optimizer.dot");
    m_executor->write_graphviz(file);
  }

  std::pair<std::shared_ptr<const Tensor>, double> Optimizer::optimize(std::shared_ptr<const Tensor> input, std::shared_ptr<const Tensor> expectedOutput, unsigned tag)
  {
    auto parameters = m_model->parameters(TAG_ALL);

    // 1: Input
    m_executor->input("input", {input});
    m_executor->input("expected output", {expectedOutput});
    m_executor->input("state", m_state);

    std::vector<std::shared_ptr<const Tensor>> tmp;
    for(std::reference_wrapper parameter : parameters)
      tmp.push_back(parameter);

    m_executor->input("parameters", std::move(tmp));

    // 2: Output
    auto output = m_executor->output("output").front();

    auto newParameters = m_executor->output("new parameters");
    {
      std::unordered_set<std::shared_ptr<const Tensor>*> trainableParametersSet;
      for(std::reference_wrapper trainableParameter : m_model->parameters(tag))
        trainableParametersSet.insert(&trainableParameter.get());

      // Only update trainable parameters
      for(size_t i=0; i<parameters.size(); ++i)
        if(trainableParametersSet.contains(&parameters[i].get()))
          parameters[i].get() = std::move(newParameters[i]);
    }

    auto newState = m_executor->output("new state");
    m_state = std::move(newState);

    // 5: Return
    // TODO: May be also put that into the computational graph
    const double cost = ((output->asVector()-expectedOutput->asVector()) * 2.0).squaredNorm();
    return std::make_pair(std::move(output), cost);
  }
}
