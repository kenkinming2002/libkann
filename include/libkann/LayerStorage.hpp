#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>
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
  };
}
