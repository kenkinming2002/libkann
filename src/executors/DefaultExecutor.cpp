#include <libkann/executors/DefaultExecutor.hpp>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/topological_sort.hpp>

#include <iterator>

namespace kann
{
  void DefaultExecutor::build()
  {
    GraphExecutor::build();

    m_ordering.reserve(boost::num_vertices(graph()));
    boost::topological_sort(graph(), std::back_inserter(m_ordering));
    std::reverse(m_ordering.begin(), m_ordering.end());

    m_data.resize(boost::num_vertices(graph()));
  }

  void DefaultExecutor::input(std::string name, std::vector<std::shared_ptr<const Tensor>> input)
  {
    if(!m_dirty)
    {
      for(auto& datum : m_data)
        datum.reset();

      m_dirty = true;
    }

    const auto& inputVertices = this->inputVertices(name);

    assert(inputVertices.size() == input.size());
    for(size_t i=0; i<input.size(); ++i)
      datum(inputVertices[i]) = std::move(input[i]);
  }

  std::vector<std::shared_ptr<const Tensor>> DefaultExecutor::output(std::string name)
  {
    if(m_dirty)
    {
      // Evaluate
      for(vertex_type vertex : m_ordering)
      {
        if(datum(vertex))
          continue; // One of the input vertex

        const Node& node = graph()[vertex];

        std::vector<const Tensor*> inputs(node.inputCount);
        for(auto [it, end] = boost::in_edges(vertex, graph()); it != end; ++it)
        {
          edge_type edge = *it;
          vertex_type inputVertex = boost::source(edge, graph());

          const Connection& connection = graph()[edge];
          inputs[connection.i] = datum(inputVertex).get();
        }

        datum(vertex) = node.op->process(std::move(inputs));
      }
      m_dirty = false;
    }

    const auto& outputVertices = this->outputVertices(name);

    std::vector<std::shared_ptr<const Tensor>> outputs(outputVertices.size());
    for(size_t i=0; i<outputVertices.size(); ++i)
      outputs[i] = datum(outputVertices[i]);

    return outputs;
  }
}

