#pragma once

#include <libkann/Tensor.hpp>
#include <libkann/Variable.hpp>

#include <vector>
#include <string>
#include <memory>

namespace kann
{
  // An executor has list of input and output variables, which may be named
  // Such collection of variables is represented by
  //
  // std::unordered_map<std::string, std::vector<std::shared_ptr<const Variable>>
  class Executor
  {
  public:
    virtual ~Executor() = default;

  public:
    virtual void addInput(std::string name, std::vector<std::shared_ptr<const Variable>> variables) = 0;
    virtual void addOutput(std::string name, std::vector<std::shared_ptr<const Variable>> variables) = 0;

  public:
    virtual void build() = 0;

  public:
    virtual void input(std::string name, std::vector<std::shared_ptr<const Tensor>> input) = 0;
    virtual std::vector<std::shared_ptr<const Tensor>> output(std::string name) = 0;

  public:
    virtual void write_graphviz(std::ostream& os) const = 0;
  };

  std::unique_ptr<Executor> makeDefaultExecutor();
  std::unique_ptr<Executor> makeThreadedExecutor();
}
