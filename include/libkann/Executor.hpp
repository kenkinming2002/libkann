#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Variable.hpp>

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
    virtual void build(std::vector<CRef<Variable>> inputs, std::vector<CRef<Variable>> outputs) = 0;
    virtual std::vector<CRef<Tensor>> process(std::vector<CRef<Tensor>> inputs) = 0;

  public:
    virtual void write_graphviz(std::ostream& os) const = 0;
  };
}
