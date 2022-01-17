#include <libkann/Predictor.hpp>

#include <iterator>

namespace kann
{
  Predictor::Predictor(std::shared_ptr<Model> model)
    : m_model(std::move(model))
  {
    std::shared_ptr<const Variable> input, output;
    std::vector<std::shared_ptr<const Variable>> parametersVariables;
    std::vector<std::shared_ptr<const Variable>> stateVariables, newStateVariables;

    // Initialize
    input = std::make_shared<const Variable>();
    parametersVariables = m_model->parametersVariables();
    stateVariables      = m_model->makeStateVariables();
    std::tie(output, newStateVariables) = (*m_model)(input, stateVariables);

    // Create executor
    std::vector<std::shared_ptr<const Variable>> inputs;
    {
      inputs.push_back(input);
      inputs.insert(inputs.end(), std::move_iterator(parametersVariables.begin()), std::move_iterator(parametersVariables.end()));
      inputs.insert(inputs.end(), std::move_iterator(stateVariables.begin()),      std::move_iterator(stateVariables.end()));
    }

    std::vector<std::shared_ptr<const Variable>> outputs;
    {
      outputs.push_back(output);
      outputs.insert(outputs.end(), std::move_iterator(newStateVariables.begin()), std::move_iterator(newStateVariables.end()));
    }

    m_executor = Executor(std::move(inputs), std::move(outputs));

    // Create initial state
    m_state = m_model->makeState();
  }

  Eigen::VectorXd Predictor::predict(Eigen::VectorXd input)
  {
    const auto parameters = m_model->parameters();

    std::vector<Tensor> inputs;
    {
      inputs.emplace_back(input);
      inputs.insert(inputs.end(), std::move_iterator(parameters.begin()), std::move_iterator(parameters.end()));
      inputs.insert(inputs.end(), m_state.begin(), m_state.end());
    }

    std::vector<Tensor> outputs = m_executor.evaluate(inputs);

    Tensor output;
    std::vector<Tensor> newState;

    output = std::move(outputs[0]);
    newState.assign(std::move_iterator(outputs.begin()+1), std::move_iterator(outputs.end()));

    m_state = std::move(newState);

    return output.asVector();
  }
}
