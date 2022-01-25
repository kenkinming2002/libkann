#pragma once

#include <libkann/executors/GraphExecutor.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <future>

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
    struct WorkItem
    {
      std::vector<vertex_type> vertices;
    };
    std::vector<WorkItem> m_workItems;

  private:
    auto& datum(vertex_type vertex) { return m_data[boost::get(boost::vertex_index, graph(), vertex)]; }

  private:
    bool m_dirty;

    struct Datum
    {
      std::vector<std::shared_future<std::shared_ptr<const Tensor>>> inputFutures;
      std::promise<std::shared_ptr<const Tensor>> promise;

      std::shared_future<std::shared_ptr<const Tensor>> future;
    };
    std::vector<Datum> m_data;
  };
}
