#pragma once

#include <libkann/Export.hpp>
#include <libkann/Variable.hpp>

#include <vector>

namespace kann
{
  struct LayerStorage
  {
  public:
    std::vector<Variable> parameters;
    std::vector<Variable> states;

  public:
    std::vector<std::shared_ptr<LayerStorage>> sub_layer_storages;

  public:
    KANN_EXPORT std::vector<const Variable*> get_parameters() const;
    KANN_EXPORT std::vector<const Variable*> get_states() const;

    KANN_EXPORT std::vector<Variable*> get_parameters();
    KANN_EXPORT std::vector<Variable*> get_states();
  };
}
