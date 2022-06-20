#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  struct LayerStorage
  {
  public:
    layer_def_t def;

    std::vector<Tensor> parameters;
    std::vector<Tensor> states;

  public:
    std::vector<std::shared_ptr<LayerStorage>> sub_layer_storages;
  };
}
