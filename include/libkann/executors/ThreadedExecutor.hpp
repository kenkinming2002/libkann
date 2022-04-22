#pragma once

#include <libkann/Executor.hpp>

#include <semaphore>
#include <latch>

namespace kann
{
  class ThreadedExecutor : public Executor
  {
  public:
    std::vector<std::shared_ptr<const Tensor>> process(std::shared_ptr<const Graph> graph, std::vector<std::shared_ptr<const Tensor>> inputs) override;

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

#if 0
  protected:
    void build() override final;
    void compute() override final;

  private:
    void process(vertex_type vertex);
    void publish(vertex_type vertex);
    void submit(vertex_type vertex);

  private:
    auto& datum(vertex_type vertex) { return m_data[boost::get(boost::vertex_index, graph(), vertex)]; }

  private:
    struct Datum
    {
      std::atomic<size_t> finishedCount;
      CRef<Tensor> value;
    };
    std::vector<Datum> m_data;

    size_t m_taskCount;
    std::optional<TaskSet> m_taskSet;
#endif
  };
}
