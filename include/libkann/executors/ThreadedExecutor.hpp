#pragma once

#include <libkann/Executor.hpp>

#include <unordered_map>

namespace kann
{
  class ThreadedExecutor : public Executor
  {
  public:
    std::vector<std::vector<tensor_t>> process(graph_t graph, std::vector<std::vector<tensor_t>> inputs) override;

  private:
    struct State
    {
      struct Datum
      {
        size_t finished_count;
        tensor_t value;
      };
      std::vector<Datum> data;
    };
    std::unordered_map<graph_t, State> m_states;
  };
}
