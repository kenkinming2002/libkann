#pragma once

#include <libkann/executors/GraphExecutor.hpp>
#include <libkann/executors/details/TaskSet.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <semaphore>
#include <latch>

namespace kann
{
  class ThreadedExecutor : public GraphExecutor
  {
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
  };
}
