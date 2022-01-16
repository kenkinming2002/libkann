#pragma once

#include <libkann/Variable.hpp>

#include <unordered_map>

namespace kann
{
  /* Map variables to gradients variables */
  std::unordered_map<std::shared_ptr<const Variable>, std::shared_ptr<const Variable>> differentiate(
    const std::vector<std::shared_ptr<const Variable>>& variables,
    const std::vector<std::shared_ptr<const Variable>>& gradients);
}
