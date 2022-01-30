#pragma once

#include <libkann/executors/GraphExecutor.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <optional>

namespace kann
{
  class DefaultExecutor : public GraphExecutor
  {
  public:
    void build() override;

  public:
    void input(std::string name, std::vector<std::shared_ptr<const Tensor>> input) override;
    std::vector<std::shared_ptr<const Tensor>> output(std::string name) override;

  private:
    std::shared_ptr<const Tensor>& datum(vertex_type vertex) { return m_data[boost::get(boost::vertex_index, graph(), vertex)]; }

  private:
    std::vector<vertex_type> m_ordering;

  private:
    bool m_dirty;
    std::vector<std::shared_ptr<const Tensor>> m_data;
  };
}
