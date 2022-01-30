#pragma once

#include <functional>

#include <queue>

#include <mutex>
#include <latch>
#include <atomic>

#include <semaphore.h>

#include <iostream>

namespace kann
{
  /* NOTE: We use posix semaphore here instead of c++20 semaphore, because the
   *       implementation in libstdc++ has bug which may result in deadlock when
   *       there are more than or equal to 3 threads operating on same semaphore
   *       */
  class TaskSet
  {
  public:
    TaskSet(size_t count)
      : m_total(count)
    {
      sem_init(&m_sem, 1, 0);
    }

    ~TaskSet()
    {
      sem_destroy(&m_sem);
    }

  public:
    /* Submit a single task to context. Could be called recursively from a
     * submitted task. */
    void submit(std::function<void(TaskSet&)> task)
    {
      {
        std::lock_guard guard(m_mutex);
        m_tasks.push(std::move(task));
      }

      sem_post(&m_sem);
    }

    /* Run tasks submitted to context. */
    void run()
    {
      for(;;)
      {
        // Decrement if non-zero, does not fail spuriously
        for(;;)
        {
          size_t total = m_total.load();
          if(total == 0)
            return;

          if(m_total.compare_exchange_weak(total, total-1))
            break;
        }

        sem_wait(&m_sem);

        std::unique_lock lock(m_mutex);

        auto task = std::move(m_tasks.front());
        m_tasks.pop();

        lock.unlock();

        task(*this);
      }
    }

  private:
    /* Total count of task to be done, which may not have been submitted
     * TODO: think of a bettern name */
    std::atomic<size_t> m_total;

    sem_t m_sem;

    std::mutex m_mutex;
    std::queue<std::function<void(TaskSet&)>> m_tasks;
  };

}
