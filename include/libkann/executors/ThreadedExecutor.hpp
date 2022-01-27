#pragma once

#include <libkann/executors/GraphExecutor.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <semaphore>

namespace kann
{
  class ThreadedExecutor : public GraphExecutor
  {
  public:
    void build() override;

  public:
    void input(std::string name, std::vector<std::shared_ptr<const Tensor>> input) override;
    std::vector<std::shared_ptr<const Tensor>> output(std::string name) override;

  private:
    void process(vertex_type vertex);
    void publish(vertex_type vertex);
    void submit(vertex_type vertex);

  private:
    auto& datum(vertex_type vertex) { return m_data[boost::get(boost::vertex_index, graph(), vertex)]; }

  private:
    bool m_dirty = false;
    std::counting_semaphore<> m_sem{0};

    struct Datum
    {
      std::atomic<size_t> finishedCount;

      std::vector<std::shared_ptr<const Tensor>> inputs;
      std::shared_ptr<const Tensor> output;
    };
    std::vector<Datum> m_data;
  };
}
