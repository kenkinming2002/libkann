#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  struct Layer
  {
  public:
    layer_def_t def;

    std::vector<Tensor> parameters;
    std::vector<Tensor> states;

  public:
    std::vector<std::shared_ptr<Layer>> sub_layers;

  public:
    void set_parameters_all(std::vector<Tensor> values);
    void set_states_all(std::vector<Tensor> values);

    std::vector<Tensor> get_parameters_all() const;
    std::vector<Tensor> get_states_all() const;
  };
}
