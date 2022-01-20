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
    std::shared_ptr<const Variable> input, output;
    std::vector<std::shared_ptr<const Variable>> parametersVariables, parametersGradientVariables, newParametersVariables;
    std::vector<std::shared_ptr<const Variable>> stateVariables, newStateVariables;

    input = std::make_shared<const Variable>();
    parametersVariables = m_model->parametersVariables(TAG_ALL);
    stateVariables      = m_model->makeStateVariables();
    std::tie(output, newStateVariables) = (*m_model)(input, stateVariables);

    const auto expectedOutput = std::make_shared<const Variable>();
    const auto tmp            = std::make_shared<const Variable>(std::vector{output,expectedOutput}, std::make_shared<SubtractOperation>());
    const auto outputGradient = std::make_shared<const Variable>(std::vector{tmp}, std::make_shared<MultiplyOperation>(2.0));

    parametersVariables = m_model->parametersVariables(TAG_ALL);
    {
      const auto gradientsMap = differentiate({output}, {outputGradient});
      std::transform(parametersVariables.begin(), parametersVariables.end(), std::back_inserter(parametersGradientVariables), [&gradientsMap](const auto& variable){
          return gradientsMap.at(variable);
      });
    }

    for(size_t i=0; i<parametersVariables.size(); ++i)
    {
      const auto tmp = std::make_shared<const Variable>(std::vector{parametersGradientVariables[i]}, std::make_shared<MultiplyOperation>(learningRate));
      const auto result = std::make_shared<const Variable>(std::vector{parametersVariables[i], std::move(tmp)}, std::make_shared<SubtractOperation>());
      newParametersVariables.push_back(std::move(result));
    }

    std::vector<std::shared_ptr<const Variable>> inputs;
    {
      inputs.push_back(input);
      inputs.push_back(expectedOutput);

      inputs.insert(inputs.end(), std::move_iterator(parametersVariables.begin()), std::move_iterator(parametersVariables.end()));
      inputs.insert(inputs.end(), std::move_iterator(stateVariables.begin()), std::move_iterator(stateVariables.end()));

    }

    std::vector<std::shared_ptr<const Variable>> outputs;
    {
      outputs.push_back(output);

      outputs.insert(outputs.end(), std::move_iterator(newParametersVariables.begin()), std::move_iterator(newParametersVariables.end()));
      outputs.insert(outputs.end(), std::move_iterator(newStateVariables.begin()), std::move_iterator(newStateVariables.end()));
    }

    m_executor = makeDefaultExecutor(std::move(inputs), std::move(outputs));

    // Create initial state
    m_state = m_model->makeState();

    std::ofstream file("output/optimizer.dot");
    m_executor->write_graphviz(file);
  }

  std::pair<std::shared_ptr<const Tensor>, double> Optimizer::optimize(std::shared_ptr<const Tensor> input, std::shared_ptr<const Tensor> expectedOutput, unsigned tag)
  {
    auto parameters = m_model->parameters(TAG_ALL);

    // 1: Input
    std::vector<std::shared_ptr<const Tensor>> inputs;
    {
      inputs.push_back(input);
      inputs.push_back(expectedOutput);

      inputs.insert(inputs.end(), m_state.begin(), m_state.end());
      inputs.insert(inputs.end(), std::move_iterator(parameters.begin()), std::move_iterator(parameters.end()));
    }

    // 2: Outputs
    // TODO: Support partial evaluation
    std::vector<std::shared_ptr<const Tensor>> outputs = m_executor->evaluate(std::move(inputs));


    // 3: Parsing the output
    size_t i = 0;

    std::shared_ptr<const Tensor> output = outputs[i++];

    std::vector<std::shared_ptr<const Tensor>> newParameters(parameters.size());
    for(auto& parameter : newParameters)
      parameter = std::move(outputs[i++]);

    std::vector<std::shared_ptr<const Tensor>> newState(m_state.size());
    for(auto& state : newState)
      state = std::move(outputs[i++]);

    // 4: Updating
    std::unordered_set<std::shared_ptr<const Tensor>*> trainableParametersSet;
    for(std::reference_wrapper trainableParameter : m_model->parameters(tag))
      trainableParametersSet.insert(&trainableParameter.get());

    // Only update trainable parameters
    for(size_t i=0; i<parameters.size(); ++i)
      if(trainableParametersSet.contains(&parameters[i].get()))
        parameters[i].get() = std::move(newParameters[i]);

    // 5: Return
    // TODO: May be also put that into the computational graph
    const double cost = ((output->asVector()-expectedOutput->asVector()) * 2.0).squaredNorm();
    return std::make_pair(std::move(output), cost);
  }
}
