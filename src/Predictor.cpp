#include <libkann/Predictor.hpp>

#include <libkann/DefaultExecutor.hpp>

#include <fstream>
#include <iterator>

namespace kann
{
  Predictor::Predictor(std::shared_ptr<Model> model)
    : m_model(std::move(model))
  {
    // 1: Create Executor
    m_executor = makeDefaultExecutor();

    auto inputVariable = std::make_shared<const Variable>();
    m_executor->addInput("input", {inputVariable});

    auto parametersVariables = m_model->parametersVariables(TAG_ALL);
    m_executor->addInput("parameters", parametersVariables);

    auto stateVariables = m_model->makeStateVariables();
    m_executor->addInput("state", stateVariables);

    auto [outputVariable, newStateVariables] = (*m_model)(inputVariable, stateVariables);
    m_executor->addOutput("output", {outputVariable});
    m_executor->addOutput("new state", newStateVariables);

    m_executor->build();

    // 2: Create initial state
    m_state = m_model->makeState();

    // 3: DEBUG
    std::ofstream file("output/optimizer.dot");
    m_executor->write_graphviz(file);
  }

  std::shared_ptr<const Tensor> Predictor::predict(std::shared_ptr<const Tensor> input)
  {
    auto parameters = m_model->parameters(TAG_ALL);

    // 1: Input
    m_executor->input("input", {input});
    m_executor->input("state", m_state);

    std::vector<std::shared_ptr<const Tensor>> tmp;
    for(std::reference_wrapper parameter : parameters)
      tmp.push_back(parameter);

    m_executor->input("parameters", std::move(tmp));

    // 2: Output
    auto output = m_executor->output("output").front();

    auto newState = m_executor->output("new state");
    m_state = std::move(newState);

    return output;
  }
}
