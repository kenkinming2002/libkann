#pragma once

#include <libkann/executors/GraphExecutor.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <optional>

namespace kann
{
  class DefaultExecutor : public GraphExecutor
  {
  protected:
    void build() override final;
    void compute() override final;

  private:
    std::vector<vertex_type> m_ordering;
  };
}
