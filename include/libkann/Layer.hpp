#pragma once

#include <libkann/Types.hpp>

#include <vector>

namespace kann
{
  struct Layer
  {
  public:
    layer_def_t def;

    std::vector<tensor_t> parameters;
    std::vector<tensor_t> states;

  public:
    std::vector<std::shared_ptr<Layer>> sub_layers;

  public:
    void set_parameters_all(std::vector<tensor_t> values);
    void set_states_all(std::vector<tensor_t> values);

    std::vector<tensor_t> get_parameters_all() const;
    std::vector<tensor_t> get_states_all() const;
  };
}
