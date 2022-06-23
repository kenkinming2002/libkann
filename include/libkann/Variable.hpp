#pragma once

#include <libkann/Tensor.hpp>

#include <optional>

namespace kann
{
  struct Variable
  {
    Tensor<float> value;
    std::optional<Tensor<float>> gradient;
    std::vector<Tensor<float>> optimizer_states;
  };
}
