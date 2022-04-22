#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Variable.hpp>

#include <libkann/Graph.hpp>

#include <vector>
#include <span>
#include <string>
#include <memory>
#include <numeric>

namespace kann
{
  class Executor
  {
  public:
    enum class Type { DEFAULT, THREADED };
    static std::unique_ptr<Executor> create(Type type);

  public:
    virtual ~Executor() = default;

  public:
    virtual std::vector<std::shared_ptr<const Tensor>> process(std::shared_ptr<const Graph> graph, std::vector<std::shared_ptr<const Tensor>> inputs) = 0;
  };
}
