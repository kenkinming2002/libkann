#pragma once

#include <libkann/Tensor.hpp>

#include <optional>

namespace kann
{
  struct Variable
  {
    Tensor value;
    std::optional<Tensor> gradient;
    std::vector<Tensor> optimizer_states;
  };
}
