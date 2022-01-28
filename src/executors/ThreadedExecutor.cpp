#include <libkann/executors/ThreadedExecutor.hpp>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/topological_sort.hpp>

#include <thread>

#include <unordered_map>
#include <queue>

#include <iterator>

#include <condition_variable>
#include <mutex>
#include <semaphore>

namespace kann
{
  /* What if thread pool is global,
   * but before using it, we have create single use light-weight context
   *
   * Then, upon context destruction, we can ensure that every function submitted
   * to the context is completed, and thus is free from lifetime issues.
   *
   * Specifically, it is should no longer possible for thread pool to contain
   * reference to destructed objects who have submitted the work.
   * */
  struct ThreadPool
  {
  public:
    ThreadPool()
    {
      unsigned count = std::thread::hardware_concurrency()-1;
      for(unsigned i=0; i<count; ++i)
        m_workers.emplace_back(std::bind(&ThreadPool::run, this, std::placeholders::_1));
    }

    ~ThreadPool()
    {
      for(auto& worker : m_workers)
        worker.request_stop();

      m_sem.release(m_workers.size());
    }

  public:
    void submit(std::function<void()> task)
    {
      {
        std::unique_lock lk(m_mutex);
        m_tasksQueue.push(std::move(task));
      }

      m_sem.release();
    }

  private:
    void run(std::stop_token stop_token)
    {
      for(;;)
      {
        m_sem.acquire();

        // Stop
        if(stop_token.stop_requested())
          break;

        // Retrive task
        std::function<void()> task;
        {
          std::unique_lock lk(m_mutex);

          task = std::move(m_tasksQueue.front());
          m_tasksQueue.pop();
        }

        // Run task
        task();
      }
    }

  private:
    std::counting_semaphore<> m_sem{0};

    std::mutex m_mutex;
    std::queue<std::function<void()>> m_tasksQueue;

    std::vector<std::jthread> m_workers;
  };

  class Context
  {
  public:
    void submit();
    void wait();
  };

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
  }

  /* We submit work when calling input and wait for them to complete when we
   * call output */

  void ThreadedExecutor::input(std::string name, std::vector<std::shared_ptr<const Tensor>> input)
  {
    if(!m_dirty)
    {
      for(auto& datum : m_data)
        datum.finishedCount.store(0);

      m_dirty = true;
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
      for(size_t i=0; i<boost::num_vertices(graph()); ++i)
        m_sem.acquire();

      m_dirty = false;
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

    m_sem.release();
  }

  void ThreadedExecutor::submit(vertex_type vertex)
  {
    static ThreadPool threadPool;
    threadPool.submit(std::bind(&ThreadedExecutor::process, this, vertex));
  }
}

