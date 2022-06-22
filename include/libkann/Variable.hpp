#pragma once

#include <libkann/Tensor.hpp>

#include <optional>

namespace kann
{
  struct Variable
  {
    MutableTensor value;
    std::optional<Tensor> gradient;
    std::vector<MutableTensor> optimizer_states;
  };
}
