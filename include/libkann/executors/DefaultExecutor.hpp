#pragma once

#include <libkann/Executor.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <optional>

namespace kann
{
  class DefaultExecutor : public Executor
  {
  public:
    std::vector<std::shared_ptr<const Tensor>> process(std::shared_ptr<const Graph> graph, std::vector<std::shared_ptr<const Tensor>> inputs) override;

  private:
    struct State
    {
      std::vector<size_t> ordering;
      std::vector<std::shared_ptr<const Tensor>> values;
    };
    std::unordered_map<std::shared_ptr<const Graph>, State> m_states;
  };
}
