#pragma once

#include <libkann/Tensor.hpp>

namespace kann
{
  struct Variable
  {
    Tensor value;
    Tensor gradient;

    std::vector<Tensor> optimizer_states;
  };
}
