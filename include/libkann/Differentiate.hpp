#pragma once

#include <libkann/Variable.hpp>

#include <unordered_map>

namespace kann
{
  /* Map variables to gradients variables */
  std::unordered_map<CRef<Variable>, CRef<Variable>> differentiate(
    const std::vector<CRef<Variable>>& variables,
    const std::vector<CRef<Variable>>& gradients);
}
