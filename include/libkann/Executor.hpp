#pragma once

#include <libkann/Types.hpp>

#include <vector>

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
    virtual std::vector<std::vector<tensor_t>> process(graph_t graph, std::vector<std::vector<tensor_t>> inputs) = 0;
  };
}
