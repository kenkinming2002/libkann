#pragma once

#include <libkann/Types.hpp>

#include <vector>
#include <unordered_map>

namespace kann
{
  /* Map variables to gradients variables */
  std::unordered_map<variable_t, variable_t> differentiate(
    const std::vector<variable_t>& variables,
    const std::vector<variable_t>& gradients);
}
