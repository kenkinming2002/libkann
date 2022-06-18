#pragma once

#include <libkann/Export.hpp>

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
    KANN_EXPORT void set_parameters_all(std::vector<Tensor> values);
    KANN_EXPORT void set_states_all(std::vector<Tensor> values);

    KANN_EXPORT std::vector<Tensor> get_parameters_all() const;
    KANN_EXPORT std::vector<Tensor> get_states_all() const;
  };
}
