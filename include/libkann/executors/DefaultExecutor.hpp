#pragma once

#include <libkann/Executor.hpp>

#include <unordered_map>

namespace kann
{
  class DefaultExecutor : public Executor
  {
  public:
    std::vector<std::vector<tensor_t>> process(graph_t graph, std::vector<std::vector<tensor_t>> inputs) override;

  private:
    struct State
    {
      std::vector<size_t> ordering;
      std::vector<tensor_t> values;
    };
    std::unordered_map<graph_t, State> m_states;
  };
}
