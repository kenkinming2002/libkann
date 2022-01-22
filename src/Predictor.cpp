#include <libkann/Predictor.hpp>

#include <libkann/DefaultExecutor.hpp>

#include <fstream>
#include <iterator>

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

  Predictor::Predictor(std::shared_ptr<const Model> model)
    : m_model(std::move(model))
  {
    auto parameters = m_model->parameters(TAG_ALL);
    auto states     = m_model->makeStates();

    // 1: Create Executor
    m_executor = makeDefaultExecutor();

    auto inputVariable = std::make_shared<const Variable>();
    m_executor->addInput("input", {inputVariable});

    auto parametersVariables = convert(parameters, [](const std::shared_ptr<const Parameter>& parameter){ return parameter->variable; });
    m_executor->addInput("parameters", parametersVariables);

    auto statesVariables = convert(states, [](const std::shared_ptr<Parameter>& parameter){ return parameter->variable; });
    m_executor->addInput("states", statesVariables);

    auto [outputVariable, newStateVariables] = (*m_model)(inputVariable, statesVariables);
    m_executor->addOutput("output", {outputVariable});
    m_executor->addOutput("new states", newStateVariables);

    m_executor->build();

    // 2: Create initial state
    auto statesValues = convert(states, [](const std::shared_ptr<Parameter>& parameter){ return parameter->value; });
    m_statesValues = std::move(statesValues);

    // 3: DEBUG
    std::ofstream file("output/predictor.dot");
    m_executor->write_graphviz(file);
  }

  std::shared_ptr<const Tensor> Predictor::predict(std::shared_ptr<const Tensor> input)
  {
    auto parameters = m_model->parameters(TAG_ALL);

    // 1: Input
    m_executor->input("input", {input});
    m_executor->input("states", m_statesValues);

    auto parametersValues = convert(parameters, [](const std::shared_ptr<const Parameter>& parameter){ return parameter->value; });
    m_executor->input("parameters", std::move(parametersValues));

    // 2: Output
    auto output = m_executor->output("output").front();

    auto newStatesValues = m_executor->output("new states");
    m_statesValues = std::move(newStatesValues);

    return output;
  }
}
