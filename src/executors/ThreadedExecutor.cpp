#include <libkann/executors/ThreadedExecutor.hpp>

#include <libkann/executors/details/ThreadPool.hpp>

#include <libkann/Operation.hpp>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/topological_sort.hpp>

#include <unordered_map>

#include <iterator>

namespace kann
{
  void ThreadedExecutor::build()
  {
    const auto count = boost::num_vertices(this->graph());

    m_data = std::vector<Datum>(count);
    auto [begin, end] = boost::vertices(graph());
    m_taskCount = std::count_if(begin, end, [this](vertex_type vertex){
      const Node& node = graph()[vertex];
      return node.op;
    });
  }

  void ThreadedExecutor::compute()
  {
    for(auto& datum : m_data)
    {
      datum.finishedCount.store(0);
      datum.value.reset();
    }

    for(auto vertex : this->inputVertices())
      publish(vertex);

    static const auto threadCount = std::thread::hardware_concurrency();
    static ThreadPool threadPool(threadCount);

    // 1: Submit tasks to thread pool
    std::vector<std::function<void()>> tasks;
    for(unsigned i=0; i<threadCount; ++i)
      tasks.emplace_back([this](){
        m_taskSet->run();
      });

    threadPool.run(std::move(tasks));
  }

  void ThreadedExecutor::process(vertex_type vertex)
  {
    const Node& node = graph()[vertex];

    std::vector<const Tensor*> inputs;
    inputs.resize(node.inputCount);
    for(auto [it, end] = boost::in_edges(vertex, graph()); it != end; ++it)
    {
      edge_type edge = *it;
      vertex_type parentVertex = boost::source(edge, graph());

      const Connection& connection = graph()[edge];
      const Datum& parentDatum = this->datum(parentVertex);

      inputs[connection.i] = parentDatum.value.get();
    }

    Datum& datum = this->datum(vertex);
    datum.value = node.op->process(std::move(inputs));
    publish(vertex);
  }

  void ThreadedExecutor::publish(vertex_type vertex)
  {
    for(auto [it, end] = boost::out_edges(vertex, graph()); it != end; ++it)
    {
      edge_type edge = *it;
      vertex_type childVertex = boost::target(edge, graph());

      const Node& childNode = graph()[childVertex];
      Datum& childDatum = this->datum(childVertex);

      if(++childDatum.finishedCount == childNode.inputCount)
        submit(childVertex);
    }
  }

  void ThreadedExecutor::submit(vertex_type vertex)
  {
    m_taskSet->submit(std::bind(&ThreadedExecutor::process, this, vertex));
  }
}

