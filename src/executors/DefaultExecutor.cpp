#include <libkann/executors/DefaultExecutor.hpp>

#include <libkann/Operation.hpp>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/topological_sort.hpp>

#include <iterator>

namespace kann
{
  void DefaultExecutor::build()
  {
    const auto count = boost::num_vertices(this->graph());

    m_ordering.reserve(count);
    boost::topological_sort(this->graph(), std::back_inserter(m_ordering));
    std::reverse(m_ordering.begin(), m_ordering.end());
  }

  void DefaultExecutor::compute()
  {
    for(const auto vertex : m_ordering)
    {
      auto& node = this->graph()[vertex];
      if(node.value)
        continue;

      std::vector<const Tensor*> inputs(node.inputCount);
      for(auto [it, end] = boost::in_edges(vertex, this->graph()); it != end; ++it)
      {
        const auto edge = *it;
        const auto in_vertex = boost::source(edge, this->graph());

        const auto& connection = this->graph()[edge];
        const auto& in_node = this->graph()[in_vertex];

        assert(in_node.value.get());
        inputs[connection.i] = in_node.value.get();
      }
      node.value = node.op->process(std::move(inputs));
    }
  }
}

