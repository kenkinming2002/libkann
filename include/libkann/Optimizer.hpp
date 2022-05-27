#pragma once

#include <libkann/Types.hpp>

#include <vector>

namespace kann
{
  class Optimizer
  {
  public:
    virtual ~Optimizer() = default;

  public:
    struct ProcessInput
    {
      variable_t parameter;
      variable_t gradient;
    };

    struct ProcessOutput
    {
      variable_t parameter;

      std::vector<variable_t> input_states;
      std::vector<variable_t> output_states;
    };

    virtual ProcessOutput process(ProcessInput input) const = 0;

  public:
    virtual std::vector<tensor_t> create_initial_states(size_t size) const { return {}; }
  };
}
