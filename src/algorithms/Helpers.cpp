#include <libkann/algorithms/Helpers.hpp>

#include <libkann/Variable.hpp>
#include <libkann/Differentiate.hpp>
#include <libkann/operations/SubtractOperation.hpp>
#include <libkann/operations/ScaleOperation.hpp>

#include <libkann/Optimizer.hpp>

#include <libkann/LayerDef.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  std::vector<variable_t> make_input_variables(size_t count)
  {
    return ranges::views::generate_n([]() { return std::make_shared<const Variable>(); }, count) | ranges::to_vector;
  }

  std::vector<variable_t> make_simple_output_gradients(std::vector<variable_t> outputs, std::vector<variable_t> expected_outputs)
  {
    // Calculate output gradients using sum of square cost function
    return ranges::views::transform(outputs, expected_outputs, [](const auto& output, const auto& expected_output) {
      auto diff = Variable::apply(SubtractOperation(), {output, expected_output});
      return Variable::apply(ScaleOperation(2.0), {std::move(diff)});
    }) | ranges::to_vector;
  }

  PredictVariablesOutput make_predict_variables(const LayerDef& layer_def, PredictVariablesInput input)
  {
    PredictVariablesOutput output;
    output.variables.reserve(input.variables.size());
    output.states.reserve(input.states.size());
    for(const auto& input_variable : input.variables)
    {
      auto process_input = LayerDef::ProcessInput{
        .variable   = input_variable,
          .parameters = input.parameters,
          .states     = input.states
      };
      auto process_output = layer_def.process(std::move(process_input));
      output.variables.push_back(process_output.variable);
      output.states = process_output.states;
    }
    return output;
  }

  OptimizeVariablesOutput make_optimize_variables(const LayerDef& layer_def, const Optimizer& optimizer, Tag tag, OptimizeVariablesInput input)
  {
    OptimizeVariablesOutput output;
    output.parameters.reserve(input.parameters.size());

    std::unordered_map<variable_t, variable_t> gradients_map = differentiate(input.outputs, input.output_gradients);
    std::vector<Tag> parameter_tags = layer_def.parameter_tags();
    for(const auto& [input_parameter, parameter_tag] : ranges::views::zip(input.parameters, parameter_tags))
    {
      if((parameter_tag & tag) != tag)
      {
        output.parameters.push_back(input_parameter);
        continue;
      }

      auto process_input = Optimizer::ProcessInput{
        .parameter = input_parameter,
        .gradient  = gradients_map.at(input_parameter)
      };
      auto process_output = optimizer.process(std::move(process_input));

      output.parameters.push_back(process_output.parameter);
      output.input_states  |= ranges::actions::push_back(process_output.input_states);
      output.output_states |= ranges::actions::push_back(process_output.output_states);
    }
    return output;
  }
}
