#include <libkann/executors/ThreadedExecutor.hpp>

#include <libkann/executors/details/ThreadPool.hpp>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/topological_sort.hpp>

#include <unordered_map>

#include <iterator>

namespace kann
{
  void ThreadedExecutor::build()
  {
    GraphExecutor::build();

    // 1: Order the vertices
    std::vector<vertex_type> ordering;
    ordering.reserve(boost::num_vertices(graph()));
    boost::topological_sort(graph(), std::back_inserter(ordering));
    std::reverse(ordering.begin(), ordering.end());

    m_dirty = false;
    m_data = std::vector<Datum>(boost::num_vertices(graph()));

    auto [begin, end] = boost::vertices(graph());
    m_taskCount = std::count_if(begin, end, [this](vertex_type vertex){
      const Node& node = graph()[vertex];
      return node.op;
    });
  }

  /* We submit work when calling input and wait for them to complete when we
   * call output */

  void ThreadedExecutor::input(std::string name, std::vector<std::shared_ptr<const Tensor>> input)
  {
    if(!m_dirty)
    {
      m_dirty = true;

      for(auto& datum : m_data)
      {
        datum.finishedCount.store(0);
        datum.value.reset();
      }

      m_taskSet.emplace(m_taskCount);
    }

    const auto& inputVertices = this->inputVertices(name);

    assert(inputVertices.size() == input.size());
    for(size_t i=0; i<input.size(); ++i)
    {
      vertex_type vertex = inputVertices[i];
      datum(vertex).value = std::move(input[i]);
      publish(vertex); // Execution may have already started here
    }
  }

  std::vector<std::shared_ptr<const Tensor>> ThreadedExecutor::output(std::string name)
  {
    if(m_dirty)
    {
      m_dirty = false;

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

    const auto& outputVertices = this->outputVertices(name);

    std::vector<std::shared_ptr<const Tensor>> outputs(outputVertices.size());
    for(size_t i=0; i<outputVertices.size(); ++i)
    {
      outputs[i] = std::move(datum(outputVertices[i]).value); // Each output could only be retrived once
      assert(outputs[i]);
    }

    return outputs;
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

