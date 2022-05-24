#pragma once

#include <libkann/Tag.hpp>
#include <libkann/Variable.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

#include <memory>
#include <vector>
#include <random>
#include <span>

#include <assert.h>
#include <stddef.h>

namespace kann
{
  struct LayerDef;
  struct Layer
  {
  public:
    std::shared_ptr<const LayerDef> def;

    std::vector<std::shared_ptr<const Tensor>> parameters;
    std::vector<std::shared_ptr<const Tensor>> states;

  public:
    std::vector<std::shared_ptr<Layer>> sub_layers;

  public:
    void set_parameters_all(std::vector<std::shared_ptr<const Tensor>> values);
    void set_states_all(std::vector<std::shared_ptr<const Tensor>> values);

    std::vector<std::shared_ptr<const Tensor>> get_parameters_all() const;
    std::vector<std::shared_ptr<const Tensor>> get_states_all() const;
  };
}
