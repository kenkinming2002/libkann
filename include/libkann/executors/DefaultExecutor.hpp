#pragma once

#include <libkann/Executor.hpp>

#include <unordered_map>

namespace kann
{
  class DefaultExecutor : public Executor
  {
  public:
    std::vector<std::vector<tensor_t>> run(const Target& target, std::vector<std::vector<tensor_t>> inputs) const override;
  };
}
