#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tag.hpp>

#include <vector>

namespace kann
{
  std::vector<variable_t> make_input_variables(size_t count);
  std::vector<variable_t> make_simple_output_gradients(std::vector<variable_t> outputs, std::vector<variable_t> expected_outputs);

  struct PredictVariablesInput
  {
    std::vector<variable_t> variables;
    std::vector<variable_t> parameters;
    std::vector<variable_t> states;
  };

  struct PredictVariablesOutput
  {
    std::vector<variable_t> variables;
    std::vector<variable_t> states;
  };

  PredictVariablesOutput make_predict_variables(const LayerDef& layer_def, PredictVariablesInput input);

  struct OptimizeVariablesInput
  {
    std::vector<variable_t> outputs;
    std::vector<variable_t> output_gradients;
    std::vector<variable_t> expected_outputs;

    std::vector<variable_t> parameters;
  };

  struct OptimizeVariablesOutput
  {
    std::vector<variable_t> parameters;
    std::vector<variable_t> input_states;
    std::vector<variable_t> output_states;
  };

  OptimizeVariablesOutput make_optimize_variables(const LayerDef& layer_def, const Optimizer& optimizer, Tag tag, OptimizeVariablesInput input);
}
