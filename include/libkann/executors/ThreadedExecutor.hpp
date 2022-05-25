#pragma once

#include <libkann/Executor.hpp>

#include <semaphore>
#include <latch>

namespace kann
{
  class ThreadedExecutor : public Executor
  {
  public:
    std::vector<std::vector<std::shared_ptr<const Tensor>>> process(std::shared_ptr<const Graph> graph, std::vector<std::vector<std::shared_ptr<const Tensor>>> inputs) override;

  private:
    struct State
    {
      struct Datum
      {
        size_t finished_count;
        std::shared_ptr<const Tensor> value;
      };
      std::vector<Datum> data;
    };
    std::unordered_map<std::shared_ptr<const Graph>, State> m_states;
  };
}
