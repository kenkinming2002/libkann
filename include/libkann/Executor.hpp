#pragma once

#include <libkann/Tensor.hpp>

#include <vector>
#include <memory>

namespace kann
{
  class Executor
  {
  public:
    virtual ~Executor() = default;

  public:
    virtual std::vector<std::shared_ptr<const Tensor>> evaluate(std::vector<std::shared_ptr<const Tensor>> inputs) = 0;

  public:
    virtual void write_graphviz(std::ostream& os) const = 0;
  };
}
