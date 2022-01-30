#pragma once

#include <functional>

#include <thread>
#include <queue>

#include <semaphore>
#include <latch>

#include <assert.h>

#include <iostream>

namespace kann
{
  class ThreadPool
  {
  public:
    ThreadPool(size_t size)
      : m_workers(size-1) {}

  public:
    void run(std::vector<std::function<void()>> tasks)
    {
      assert(m_workers.size()+1 == tasks.size());

      std::latch latch(m_workers.size());
      for(size_t i=0; i<m_workers.size(); ++i)
        m_workers[i].run(std::move(tasks[i]), latch);

      tasks.back()();

      latch.wait();
    }

  private:
    class Worker
    {
    public:
      Worker()
        : m_sem(0), m_latch(nullptr),
          m_thread(std::bind(&Worker::loop, this, std::placeholders::_1)) {}

    public:
      void run(std::function<void()> task, std::latch& latch)
      {
        m_task  = std::move(task);
        m_latch = &latch;

        m_sem.release();
      }

    private:
      void loop(std::stop_token token)
      {
        std::stop_callback cb(token, [this](){ m_sem.release(); });

        /* Check again to see if stop has been requested before we successfully
         * register stop_callback. Possible but should practically not happen.
         * */
        if(token.stop_requested())
          return;

        for(;;)
        {
          m_sem.acquire();

          if(token.stop_requested())
            return;

          m_task();
          m_latch->count_down();
        }
      }

    private:
      std::binary_semaphore m_sem;
      std::function<void()> m_task;
      std::latch* m_latch;

      std::jthread m_thread;
    };

  private:
    std::vector<Worker> m_workers;
  };
}
