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
  static auto move_to_vector(std::span<T> data)
  {
    return std::vector(std::move_iterator(data.begin()), std::move_iterator(data.end()));
  }

  Model::Model(std::shared_ptr<Layer> layer) : m_layer(std::move(layer)) {}

  void Model::compile(size_t batch_size, std::shared_ptr<const Optimizer> optimizer, std::vector<Tag> tags)
  {
    // For book-keeping and debugging purposes only
    m_batch_size = batch_size;
    m_optimizer  = optimizer;
    m_tags = tags;

    // Predict
    {
      const size_t parameters_count = m_layer->parameters_count();
      const size_t states_count     = m_layer->states_count();

      std::shared_ptr<const Variable> input;
      std::shared_ptr<const Variable> output;

      std::vector<std::shared_ptr<const Variable>> parameters;
      std::vector<std::shared_ptr<const Variable>> input_states;
      std::vector<std::shared_ptr<const Variable>> output_states;

      input = std::make_shared<const Variable>();
      for(size_t i=0; i<parameters_count; ++i) parameters.push_back(std::make_shared<const Variable>());
      for(size_t i=0; i<states_count; ++i)     input_states.push_back(std::make_shared<const Variable>());

      Layer::ProcessInput process_input;
      process_input.variable   = input;
      process_input.parameters = parameters;
      process_input.states     = input_states;

      Layer::ProcessOutput process_output = m_layer->process(std::move(process_input));
      output        = std::move(process_output.variable);
      output_states = std::move(process_output.states);

      // Process
      auto executor_inputs  = join(std::array{std::move(input)},  std::move(parameters), std::move(input_states));
      auto executor_outputs = join(std::array{std::move(output)}, std::move(output_states));

      m_predict_executor = Executor::create(Executor::Type::DEFAULT);
      m_predict_executor->build(std::move(executor_inputs), std::move(executor_outputs));
    }

    // Optimize
    for(const auto tag : tags)
    {
      auto& optimize_executor = m_optimize_executors[tag];
      auto& optimize_states   = m_optimize_states[tag];

      const size_t parameters_count = m_layer->parameters_count();
      const size_t states_count     = m_layer->states_count();

      const std::vector<size_t> parameter_sizes = m_layer->parameter_sizes();

      std::vector<std::shared_ptr<const Variable>> input_parameters;
      std::vector<std::shared_ptr<const Variable>> output_parameters;
      std::vector<std::shared_ptr<const Variable>> input_states;
      std::vector<std::shared_ptr<const Variable>> output_states;

      for(size_t i=0; i<parameters_count; ++i) input_parameters.push_back(std::make_shared<const Variable>());
      for(size_t i=0; i<states_count; ++i)     input_states.push_back(std::make_shared<const Variable>());

      std::vector<std::shared_ptr<const Variable>> inputs;
      std::vector<std::shared_ptr<const Variable>> outputs;
      std::vector<std::shared_ptr<const Variable>> expected_outputs;
      std::vector<std::shared_ptr<const Variable>> output_gradients;

      output_states = input_states;
      for(size_t i=0; i<m_batch_size; ++i)
      {
        std::shared_ptr<const Variable> input;
        std::shared_ptr<const Variable> output;
        std::shared_ptr<const Variable> expected_output;
        std::shared_ptr<const Variable> output_gradient;

        input           = std::make_shared<const Variable>();
        expected_output = std::make_shared<const Variable>();

        Layer::ProcessInput process_input;
        process_input.variable   = input;
        process_input.parameters = input_parameters;
        process_input.states     = output_states;

        Layer::ProcessOutput process_output = m_layer->process(std::move(process_input));
        output        = std::move(process_output.variable);
        output_states = std::move(process_output.states);

        output_gradient   = Variable::apply(MultiplyOperation(2.0), {Variable::apply(SubtractOperation(), {output, expected_output})});

        inputs.push_back(std::move(input));
        outputs.push_back(std::move(output));
        expected_outputs.push_back(std::move(expected_output));
        output_gradients.push_back(std::move(output_gradient));
      }

      // Gradients
      auto gradients_map = differentiate(outputs, output_gradients);

      std::vector<std::shared_ptr<const Variable>> optimizer_input_states;
      std::vector<std::shared_ptr<const Variable>> optimizer_output_states;
      for(size_t i=0; i<parameters_count; ++i)
      {
        auto input_parameter = input_parameters[i];
        Optimizer::ProcessInput process_input;
        process_input.size      = parameter_sizes[i];
        process_input.parameter = input_parameters[i];
        process_input.gradient  = gradients_map.at(process_input.parameter);

        Optimizer::ProcessOutput process_output = optimizer->process(std::move(process_input));
        output_parameters.push_back(process_output.parameter);
        optimize_states.insert(optimize_states.end(), process_output.initial_states.begin(), process_output.initial_states.end());
        optimizer_input_states.insert(optimizer_input_states.end(), process_output.input_states.begin(), process_output.input_states.end());
        optimizer_output_states.insert(optimizer_output_states.end(), process_output.output_states.begin(), process_output.output_states.end());
      }

      auto executor_inputs  = join(std::move(inputs), std::move(expected_outputs), std::move(input_parameters), std::move(input_states), std::move(optimizer_input_states));
      auto executor_outputs = join(std::move(outputs), std::move(output_parameters), std::move(output_states), std::move(optimizer_output_states));

      optimize_executor = Executor::create(Executor::Type::DEFAULT);
      optimize_executor->build(std::move(executor_inputs), std::move(executor_outputs));

      std::ofstream file("/tmp/test.dot");
      optimize_executor->write_graphviz(file);
    }
  }

  void Model::randomize(std::default_random_engine& engine)
  {
    m_layer->randomize(engine);
  }

  // input: input, parameters, states,
  // output: output, states
  std::shared_ptr<const Tensor> Model::predict(std::shared_ptr<const Tensor> input)
  {
    auto parameters = m_layer->get_parameters();
    auto states     = m_layer->get_states();

    auto executor_inputs  = join(std::array{input}, parameters, states);
    auto executor_outputs = m_predict_executor->process(std::move(executor_inputs));
    auto [output, new_states] = split(executor_outputs, 1, states.size());

    assert(std::all_of(new_states.begin(), new_states.end(), [](const auto& v) { return (bool)v; }));

    m_layer->set_states(move_to_vector(new_states));
    return std::move(output.front());
  }

  // input: inputs, expected outputs, parameters, states, optimizer states
  // output: outputs, parameters, states, optimizer states
  auto Model::optimize(std::vector<std::shared_ptr<const Tensor>> inputs, std::vector<std::shared_ptr<const Tensor>> expected_outputs, Tag tag)
    -> std::pair<std::vector<std::shared_ptr<const Tensor>>, std::vector<double>>
  {
    auto& optimize_executor = m_optimize_executors.at(tag);
    auto& optimize_states   = m_optimize_states.at(tag);

    const size_t batch_size = inputs.size();

    auto parameters = m_layer->get_parameters();
    auto states     = m_layer->get_states();

    auto executor_inputs  = join(inputs, expected_outputs, parameters, states, optimize_states);
    auto executor_outputs = optimize_executor->process(std::move(executor_inputs));
    auto [outputs, new_parameters, new_states, new_optimizer_state_values] = split(executor_outputs, batch_size, parameters.size(), states.size(), optimize_states.size());

    m_layer->set_parameters(move_to_vector(new_parameters));
    m_layer->set_states(move_to_vector(new_states));

    optimize_states.assign(std::move_iterator(new_optimizer_state_values.begin()), std::move_iterator(new_optimizer_state_values.end()));

    std::vector<double> costs;
    for(size_t i=0; i<batch_size; ++i)
      costs.push_back((outputs[i]->asVector() - expected_outputs[i]->asVector()).squaredNorm());

    return {
      std::vector(std::move_iterator(outputs.begin()), std::move_iterator(outputs.end())),
      std::move(costs)
    };
  }
}
