#pragma once

#include <libkann/Export.hpp>

#include <libkann/Executor.hpp>

namespace kann
{
  class KANN_EXPORT DefaultExecutor : public Executor
  {
  public:
    KANN_EXPORT std::vector<std::vector<Tensor>> run(const Target& target, std::vector<std::vector<Tensor>> inputs) const override;
  };
}
