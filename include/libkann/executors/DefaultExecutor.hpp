#pragma once

#include <libkann/Executor.hpp>

namespace kann
{
  class DefaultExecutor : public Executor
  {
  public:
    std::vector<std::vector<Tensor>> run(const Target& target, std::vector<std::vector<Tensor>> inputs) const override;
  };
}
