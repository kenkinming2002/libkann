#include <libkann/Model.hpp>

#include <libkann/Differentiate.hpp>

#include <libkann/Layer.hpp>

#include <libkann/operations/SubtractOperation.hpp>
#include <libkann/operations/MultiplyOperation.hpp>

#include <fstream>
#include <random>
#include <numeric>
#include <ranges>

namespace kann
{
  template<typename... Args>
  static auto join(Args... args)
  {
    using T = std::common_type_t<decltype(args.front())...>;

    size_t total_size = 0;
    ((total_size += args.size()), ...);

    std::vector<T> result;
    (result.insert(result.end(), std::move_iterator(args.begin()), std::move_iterator(args.end())), ...);

    return result;
  }

  template<typename T, typename... Args>
  static auto split(std::span<T> data, Args... args)
  {
    std::array<std::span<T>, sizeof...(args)> result;

    size_t i = 0;
    ((assert(data.size() >= args), result[i++] = data.first(args), data = data.subspan(args)), ...);

    return result;
  }

  template<typename T, typename... Args>
  static auto split(std::vector<T>& data, Args... args)
  {
    return split(std::span(data), args...);
  }

  template<typename T>
  auto create_vector(const std::vector<T>& attributes)
  {
    std::vector<std::shared_ptr<const Variable>> vector;
    vector.reserve(attributes.size());
    for(const T& attribute : attributes)
    {
      (void)attribute;
      vector.push_back(std::make_shared<const Variable>());
    }

    return vector;
  }

  template<typename T>
  auto vector_to_map(const std::vector<T>& attributes, const std::vector<std::shared_ptr<const Variable>>& vector)
  {
    std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> map;

    size_t i = 0;
    for(const T& attribute : attributes)
    {
      auto variable = vector[i++];
      map.emplace(std::pair{attribute.layer, attribute.name}, std::move(variable));
    }

    return map;
  }

  template<typename T>
  auto map_to_vector(const std::vector<T>& attributes, const std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>>& map)
  {
    std::vector<std::shared_ptr<const Variable>> vector;
    vector.reserve(attributes.size());
    for(const T& attribute : attributes)
    {
      auto variable = map.at(std::pair{attribute.layer, attribute.name});
      vector.push_back(std::move(variable));
    }

    return vector;
  }

  Model::Model(CRef<Layer> layer)
    : m_layer(std::move(layer))
  {
    m_parameters = m_layer->parameters();
    m_states     = m_layer->states();
  }

  void Model::randomize(std::default_random_engine& engine)
  {
    for(size_t i=0; i<m_parameters.size(); ++i)
    {
      auto value = std::make_shared<Tensor>(m_parameters[i].size);
      auto dist = std::normal_distribution(m_parameters[i].mean, m_parameters[i].stddev);
      for(size_t i=0; i<value->size(); ++i)
        value->asArray()(i) = dist(engine);

      m_parameter_values.push_back(value);
    }

    for(size_t i=0; i<m_states.size(); ++i)
    {
      auto value = std::make_shared<Tensor>(m_states[i].size);
      value->asArray().setZero();

      m_state_values.push_back(value);
    }
  }

  // input: input, parameters, states,
  // output: output, states
  CRef<Tensor> Model::predict(CRef<Tensor> input)
  {
    auto& executor = this->predictExecutor();

    auto executor_inputs  = join(std::array{input}, m_parameter_values, m_state_values);
    auto executor_outputs = executor.process(std::move(executor_inputs));
    auto [output, new_state_values] = split(executor_outputs, 1, m_states.size());

    m_state_values.assign(std::move_iterator(new_state_values.begin()), std::move_iterator(new_state_values.end()));
    return std::move(output.front());
  }

  // input: inputs, expected outputs, parameters, states, optimizer states
  // output: outputs, parameters, states, optimizer states
  std::pair<std::vector<CRef<Tensor>>, std::vector<double>> Model::optimize(CRef<Optimizer> optimizer, Tag tag, std::vector<CRef<Tensor>> inputs, std::vector<CRef<Tensor>> expectedOutputs)
  {
    const size_t batch_size = inputs.size();

    auto& optimize_state = this->optimizeState(optimizer, tag, batch_size);

    auto executor_inputs  = join(inputs, expectedOutputs, m_parameter_values, m_state_values, optimize_state.values);
    auto executor_outputs = optimize_state.executor->process(std::move(executor_inputs));
    auto [outputs, new_parameter_values, new_state_values, new_optimizer_state_values] = split(executor_outputs, batch_size, m_parameters.size(), m_states.size(), optimize_state.values.size());

    m_parameter_values.assign(std::move_iterator(new_parameter_values.begin()), std::move_iterator(new_parameter_values.end()));
    m_state_values.assign(std::move_iterator(new_state_values.begin()), std::move_iterator(new_state_values.end()));
    optimize_state.values.assign(std::move_iterator(new_optimizer_state_values.begin()), std::move_iterator(new_optimizer_state_values.end()));

    std::vector<double> costs;
    for(size_t i=0; i<batch_size; ++i)
      costs.push_back((outputs[i]->asVector() - expectedOutputs[i]->asVector()).squaredNorm());

    return {
      std::vector(std::move_iterator(outputs.begin()), std::move_iterator(outputs.end())),
      std::move(costs)
    };
  }

  Executor& Model::predictExecutor()
  {
    if(!m_predictExecutor)
    {
      std::shared_ptr<const Variable> input_variable;
      std::shared_ptr<const Variable> output_variable;

      std::vector<std::shared_ptr<const Variable>> parameter_variables;
      std::vector<std::shared_ptr<const Variable>> input_state_variables;
      std::vector<std::shared_ptr<const Variable>> output_state_variables;

      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> parameter_variables_map;
      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> input_state_variables_map;
      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> output_state_variables_map;

      // Input
      input_variable = std::make_shared<const Variable>();
      parameter_variables = create_vector(m_parameters);
      input_state_variables = create_vector(m_states);

      parameter_variables_map   = vector_to_map(m_parameters, parameter_variables);
      input_state_variables_map = vector_to_map(m_states, input_state_variables);

      // Process
      auto process_input = Layer::ProcessInput{
        .variable = input_variable,
        .parameters = parameter_variables_map,
        .states     = input_state_variables_map
      };
      auto process_output = m_layer->process(process_input);
      output_variable            = process_output.variable;
      output_state_variables_map = process_output.states;

      // Output
      output_state_variables = map_to_vector(m_states, output_state_variables_map);

      auto inputs  = join(std::array{input_variable}, std::move(parameter_variables), std::move(input_state_variables));
      auto outputs = join(std::array{output_variable}, std::move(output_state_variables));

      m_predictExecutor = Executor::create(Executor::Type::DEFAULT);
      m_predictExecutor->build(std::move(inputs), std::move(outputs));
    }

    return *m_predictExecutor;
  }

  Model::OptimizeState& Model::optimizeState(CRef<Optimizer> optimizer, Tag tag, size_t batchSize)
  {
    auto config = OptimizeConfig{
      .optimizer = optimizer,
      .tag       = tag,
      .batchSize = batchSize
    };
    OptimizeState& optimize_state = m_optimizeStates[config];
    if(!optimize_state.executor)
    {
      std::vector<std::shared_ptr<const Variable>> input_variables;
      std::vector<std::shared_ptr<const Variable>> output_variables;
      std::vector<std::shared_ptr<const Variable>> expected_output_variables;
      std::vector<std::shared_ptr<const Variable>> output_gradient_variables;

      std::vector<std::shared_ptr<const Variable>> input_parameter_variables;
      std::vector<std::shared_ptr<const Variable>> output_parameter_variables;
      std::vector<std::shared_ptr<const Variable>> input_state_variables;
      std::vector<std::shared_ptr<const Variable>> output_state_variables;

      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> input_parameter_variables_map;
      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> output_parameter_variables_map;
      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> input_state_variables_map;
      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> output_state_variables_map;

      std::vector<std::shared_ptr<const Variable>> optimizer_input_state_variables;
      std::vector<std::shared_ptr<const Variable>> optimizer_output_state_variables;

      // Input Parameters
      input_parameter_variables = create_vector(m_parameters);
      input_state_variables     = create_vector(m_states);

      input_parameter_variables_map = vector_to_map(m_parameters, input_parameter_variables);
      input_state_variables_map     = vector_to_map(m_states,     input_state_variables);

      // Process
      output_state_variables_map = input_state_variables_map;
      for(size_t i=0; i<batchSize; ++i)
      {
        std::shared_ptr<const Variable> input_variable;
        std::shared_ptr<const Variable> output_variable;
        std::shared_ptr<const Variable> expected_output_variable;
        std::shared_ptr<const Variable> output_gradient_variable;

        input_variable           = std::make_shared<Variable>();
        expected_output_variable = std::make_shared<Variable>();

        auto process_input = Layer::ProcessInput{
          .variable = input_variable,
          .parameters = input_parameter_variables_map,
          .states     = output_state_variables_map
        };
        auto process_output = m_layer->process(process_input);
        output_variable            = process_output.variable;
        output_state_variables_map = process_output.states;

        output_gradient_variable   = Variable::apply(MultiplyOperation(2.0), {Variable::apply(SubtractOperation(), {output_variable, expected_output_variable})});

        input_variables.push_back(std::move(input_variable));
        output_variables.push_back(std::move(output_variable));
        expected_output_variables.push_back(std::move(expected_output_variable));
        output_gradient_variables.push_back(std::move(output_gradient_variable));
      }

      output_state_variables = map_to_vector(m_states, output_state_variables_map);

      // Gradients
      auto gradients_map = differentiate(output_variables, output_gradient_variables);

      // Optimizer states
      for(const auto& parameter : m_parameters)
      {
        auto input_parameter_variable = input_parameter_variables_map.at({parameter.layer, parameter.name});
        auto process_input = Optimizer::ProcessInput{
          .size = parameter.size,
          .parameter = input_parameter_variable,
          .gradient  = gradients_map.at(input_parameter_variable)
        };
        auto process_output = optimizer->process(process_input);
        output_parameter_variables.push_back(process_output.parameter);

        optimize_state.values.insert(optimize_state.values.end(), process_output.initial_states.begin(), process_output.initial_states.end());
        optimizer_input_state_variables.insert(optimizer_input_state_variables.end(), process_output.input_states.begin(), process_output.input_states.end());
        optimizer_output_state_variables.insert(optimizer_output_state_variables.end(), process_output.output_states.begin(), process_output.output_states.end());
      }

      auto inputs  = join(std::move(input_variables), std::move(expected_output_variables), std::move(input_parameter_variables), std::move(input_state_variables), std::move(optimizer_input_state_variables));
      auto outputs = join(std::move(output_variables), std::move(output_parameter_variables), std::move(output_state_variables), std::move(optimizer_output_state_variables));

      optimize_state.executor = Executor::create(Executor::Type::DEFAULT);
      optimize_state.executor->build(std::move(inputs), std::move(outputs));
    }

    return optimize_state;
  }

  Ref<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate)
  {
    // They have to have the same underlying structure for cross to work
    assert(lhs.m_layer.get() == rhs.m_layer.get());

    auto result = std::make_shared<Model>(lhs.m_layer);
    for(size_t i=0; i<result->m_parameters.size(); ++i)
    {
      std::normal_distribution distWeight(result->m_parameters[i].mean, result->m_parameters[i].stddev);
      std::uniform_real_distribution distMutation(0.0,1.0);
      std::uniform_int_distribution distSelection(0,1);

      auto new_value = Tensor::binaryExpr(*lhs.m_parameter_values[i], *rhs.m_parameter_values[i], [&](double a, double b){
        if(distMutation(engine)>=mutationRate)
          return distWeight(engine);
        else
          return distSelection(engine) == 0 ? a : b;
      });
      result->m_parameter_values.push_back(std::make_shared<Tensor>(std::move(new_value)));
    }

    for(size_t i=0; i<result->m_states.size(); ++i)
    {
      auto value = std::make_shared<Tensor>(result->m_states[i].size);
      value->asArray().setZero();

      result->m_state_values.push_back(value);
    }

    return result;
  }
}
