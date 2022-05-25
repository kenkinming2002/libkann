#include <libkann/executors/ThreadedExecutor.hpp>

#include <libkann/Operation.hpp>

#include <range/v3/all.hpp>

#include <unordered_map>
#include <queue>

#include <mutex>
#include <future>
#include <barrier>
#include <thread>

namespace kann
{
  /* NOTE: We use posix semaphore here instead of c++20 semaphore, because the
   *       implementation in libstdc++ has bug which may result in deadlock when
   *       there are more than or equal to 3 threads operating on same semaphore
   *       */
  template<typename Task>
  class TaskSet
  {
  public:
    TaskSet(size_t count) : m_count(count), m_tickets(0) { sem_init(&m_sem, 0, 0); }
    ~TaskSet() { sem_destroy(&m_sem); }

  public:
    void submit(std::span<Task> tasks)
    {
      std::unique_lock<std::mutex> lk(m_mutex);
      for(Task& task : tasks)
        m_tasks.push(std::move(task));

      lk.unlock();

      for(size_t i=0; i<tasks.size(); ++i)
        sem_post(&m_sem);
    }

    template<typename Handler>
    void run(Handler handler) requires(std::is_invocable_v<Handler, const Task&>)
    {
      while(m_tickets.fetch_add(1, std::memory_order_relaxed) < m_count)
      {
        sem_wait(&m_sem);

        std::unique_lock<std::mutex> lk(m_mutex);
        Task task = std::move(m_tasks.front());
        m_tasks.pop();

        lk.unlock();

        handler(task);
      }
    }

  private:
    size_t m_count;

    std::mutex m_mutex;
    std::queue<Task> m_tasks;

    std::atomic<size_t> m_tickets; // Number of total tasks to run
    sem_t m_sem;
  };

  class ThreadPool
  {
  public:
    ThreadPool(size_t size)
      : m_barrier(size),
        m_workers(size-1)
    {
      for(size_t i=0; i<m_workers.size(); ++i)
        m_workers[i].thread = std::jthread(std::bind(&ThreadPool::loop, this, i, std::placeholders::_1));
    }

  public:
    void run(std::vector<std::function<void()>> tasks)
    {
      assert(m_workers.size()+1 == tasks.size());
      for(size_t i=0; i<m_workers.size(); ++i)
      {
        Worker& worker = m_workers[i];

        std::unique_lock lk(worker.mutex);
        worker.task = std::move(tasks[i]);
        worker.cv.notify_one();
        lk.unlock();
      }

      tasks.back()();
      m_barrier.arrive_and_wait();
    }

  private:
    void loop(size_t i, std::stop_token token)
    {
      Worker& worker = m_workers[i];
      for(;;)
      {
        std::unique_lock lk(worker.mutex);
        worker.cv.wait(lk, token, [&]() { return (bool)worker.task; });
        if(token.stop_requested())
          return;

        auto task = std::exchange(worker.task, nullptr);
        lk.unlock();

        task();
        m_barrier.arrive_and_wait();
      }
    }

  private:
    struct Worker
    {
      std::mutex mutex;
      std::condition_variable_any cv;

      std::function<void()> task;
      std::jthread thread;
    };

    std::barrier<> m_barrier;
    std::vector<Worker> m_workers;
  };

  struct Task
  {
    size_t index;
  };

  std::vector<std::vector<std::shared_ptr<const Tensor>>> ThreadedExecutor::process(std::shared_ptr<const Graph> graph, std::vector<std::vector<std::shared_ptr<const Tensor>>> inputs)
  {
    auto it = m_states.find(graph);
    if(it == m_states.end())
    {
      State state;
      state.data.resize(graph->size());
      it = m_states.emplace(graph, std::move(state)).first;
    }

    State& state = it->second;

    // 2: Reset
    std::fill(state.data.begin(), state.data.end(), State::Datum{.finished_count = 0, .value = nullptr});

    // 3: Task set
    TaskSet<Task> task_set(state.data.size());

    // 4: Inputs
    {
      std::vector<Task> new_tasks;
      for(const auto& [sub_input_indices, sub_inputs] : ranges::views::zip(graph->input_indices(), inputs))
        for(const auto& [input_index, input] : ranges::views::zip(sub_input_indices, sub_inputs))
        {
          state.data[input_index].value = input;
          new_tasks.push_back(Task{.index = input_index});
        }

      task_set.submit(new_tasks);
    }

    // 5: Compute
    const auto& nodes = graph->nodes();
    auto handler = [&](const Task& task)
    {
      const auto& node = nodes[task.index];
      if(!state.data[task.index].value)
      {
        // Process
        auto inputs = node.input_indices | ranges::views::transform([&](size_t input_index) {
            assert(state.data[input_index].value);
            return state.data[input_index].value.get();
        }) | ranges::to_vector;
        assert(node.op);

        state.data[task.index].value = node.op->process(std::move(inputs));
      }

      // Publish
      std::vector<Task> new_tasks;
      for(size_t output_index : node.output_indices)
      {
        const auto& output_node = nodes[output_index];
        if(++std::atomic_ref(state.data[output_index].finished_count) == output_node.input_indices.size())
          new_tasks.push_back(Task{.index = output_index});
      }
      task_set.submit(new_tasks);
    };

    static ThreadPool thread_pool(std::thread::hardware_concurrency());

    std::vector<std::function<void()>> tasks(std::thread::hardware_concurrency(), [&]() { task_set.run(handler); });
    thread_pool.run(std::move(tasks));

    // 6: Outputs
    return graph->output_indices() | ranges::views::transform([&](const auto& sub_output_indices) {
        return sub_output_indices | ranges::views::transform([&](size_t output_index) {
            return state.data[output_index].value;
        }) | ranges::to_vector;
    }) | ranges::to_vector;
  }
}

