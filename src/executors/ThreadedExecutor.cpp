#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/subgraph.hpp>
#include <future>
#include <libkann/executors/ThreadedExecutor.hpp>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/topological_sort.hpp>

#include <condition_variable>
#include <iterator>
#include <mutex>
#include <thread>

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

    // 2: Remove input vertices
    ordering.erase(std::remove_if(ordering.begin(), ordering.end(), [this](vertex_type vertex){
      const Node& node = graph()[vertex];
      return !node.op;
    }), ordering.end());

    m_data.resize(boost::num_vertices(graph()));

    // 3: Distribute vertex into work items
    m_workItems.resize(std::thread::hardware_concurrency());
    for(vertex_type vertex : ordering)
    {
      // Prioritize distributing if it follows immediately
      size_t size = SIZE_MAX;
      WorkItem* result = nullptr;

      // 3.1: Look for consecutive vertex
      for(auto [it, end] = boost::in_edges(vertex, graph()); it != end; ++it)
      {
        vertex_type parentVertex = boost::source(*it, graph());
        for(auto& workItem : m_workItems)
          if(!workItem.vertices.empty() && parentVertex == workItem.vertices.back())
            if(workItem.vertices.size() < size)
            {
              size = workItem.vertices.size();
              result = &workItem;
            }
      }

      if(result)
        goto end;

      // 3.2: If unsuccessful look for smallest work item
      for(auto& workItem : m_workItems)
        if(workItem.vertices.size() < size)
        {
          size = workItem.vertices.size();
          result = &workItem;
        }

      if(result)
        goto end;

      assert(false && "Unreachable");
end:
      assert(result);
      result->vertices.push_back(vertex);
    }

    for(const auto& workItem : m_workItems)
    {
      std::cout << "WorkItem:";
      for(vertex_type vertex : workItem.vertices)
        std::cout << vertex << ", ";

      std::cout << '\n';
    }
  }

  void ThreadedExecutor::input(std::string name, std::vector<std::shared_ptr<const Tensor>> input)
  {
    if(!m_dirty)
    {
      for(auto& datum : m_data)
      {
        datum.promise = std::promise<std::shared_ptr<const Tensor>>();
        datum.future = datum.promise.get_future().share();
        assert(datum.future.valid());
      }

      for(auto [it1, end1] = boost::vertices(graph()); it1 != end1; ++it1)
      {
        vertex_type vertex = *it1;

        const Node& node = graph()[vertex];
        Datum& datum = this->datum(vertex);

        datum.inputFutures.resize(node.inputCount);
        for(auto [it2, end2] = boost::in_edges(vertex, graph()); it2 != end2; ++it2)
        {
          edge_type edge = *it2;
          vertex_type parentVertex = boost::source(edge, graph());

          const Connection& connection = graph()[edge];
          Datum& parentDatum = this->datum(parentVertex);

          assert(parentDatum.future.valid());
          datum.inputFutures[connection.i] = parentDatum.future;
        }
      }

      m_dirty = true;
    }

    const auto& inputVertices = this->inputVertices(name);

    assert(inputVertices.size() == input.size());
    for(size_t i=0; i<input.size(); ++i)
      datum(inputVertices[i]).promise.set_value(std::move(input[i]));
  }

  std::vector<std::shared_ptr<const Tensor>> ThreadedExecutor::output(std::string name)
  {
    if(m_dirty)
    {
      std::vector<std::future<void>> futures;
      for(const auto& workItem : m_workItems)
        futures.push_back(std::async(std::launch::async, [this, &workItem](){
          for(vertex_type vertex : workItem.vertices)
          {
            const Node& node = graph()[vertex];
            Datum& datum = this->datum(vertex);

            if(!node.op)
              continue;

            std::vector<std::shared_ptr<const Tensor>> inputs;
            inputs.reserve(node.inputCount);
            for(auto& inputFuture : datum.inputFutures)
              inputs.push_back(inputFuture.get());

            datum.promise.set_value(node.op->process(std::move(inputs)));
          }
        }));

      for(auto& future : futures)
        future.get();

      m_dirty = false;
    }

    const auto& outputVertices = this->outputVertices(name);

    std::vector<std::shared_ptr<const Tensor>> outputs(outputVertices.size());
    for(size_t i=0; i<outputVertices.size(); ++i)
      outputs[i] = datum(outputVertices[i]).future.get(); // Each output could only be retrived once

    return outputs;
  }
}

