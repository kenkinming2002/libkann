#include <libkann/Optimizer.hpp>

namespace kann
{
  void Optimizer::Info::add_state(Tensor initial, size_t input_index, size_t output_index)
  {
    initial_states.push_back(std::move(initial));
    input_states_indices.push_back(input_index);
    output_states_indices.push_back(output_index);
  }
}
