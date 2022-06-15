#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Graph.hpp>

#include <vector>

namespace kann
{
  class Executor
  {
  public:
    struct Target
    {
      Graph graph;
      std::vector<std::vector<size_t>> input_indices;
      std::vector<std::vector<size_t>> output_indices;
    };

  public:
    enum class Type { DEFAULT, THREADED };
    static std::unique_ptr<Executor> create(Type type);

  public:
    virtual ~Executor() = default;

  public:
    virtual std::vector<std::vector<Tensor>> run(const Target& target, std::vector<std::vector<Tensor>> inputs) const = 0;
  };
}
