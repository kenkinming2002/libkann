#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  class Optimizer
  {
  public:
    virtual ~Optimizer() = default;

  public:
    struct Info
    {
      std::vector<Tensor> initial_states;
      std::vector<size_t> input_states_indices;
      std::vector<size_t> output_states_indices;

      void add_state(Tensor initial, size_t input_index, size_t output_index)
      {
        initial_states.push_back(std::move(initial));
        input_states_indices.push_back(input_index);
        output_states_indices.push_back(output_index);
      }
    };

  public:
    virtual size_t process(Graph& graph, Info& info, Shape shape, size_t index, size_t gradient_index) const = 0;
  };
}
