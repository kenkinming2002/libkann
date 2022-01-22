#pragma once

#include <coroutine>

namespace kann
{
  /* A coroutine task type, which returns result ultimately, but in the process
   * of its execution, may update its caller by return Info structure */
  template<typename Result, typename Info>
  struct Task
  {
  public:
    struct promise_type;
    typedef std::coroutine_handle<promise_type> handle_type;

  public:
    struct promise_type
    {
    public:
      Task get_return_object()
      {
        return Task(handle_type::from_promise(*this));
      }

      auto initial_suspend() noexcept
      {
        return std::suspend_always{};
      }

      auto final_suspend() noexcept
      {
        return std::suspend_always{};
      }

      auto yield_value(Info info)
      {
        this->info = std::move(info);
        return std::suspend_always{};
      }

      void return_value(Result result)
      {
        this->result = std::move(result);
      }

      void unhandled_exception() {}

    public:
      Info info;
      Result result;
    };


  public:
    explicit Task(handle_type handle) : m_handle(handle) {}
    ~Task()
    {
      if(m_handle)
        m_handle.destroy();
    }

  public:
    bool done() const
    {
      return m_handle.done();
    }

    bool step()
    {
      m_handle.resume();
      return m_handle.done();
    }

    Info& info()
    {
      return m_handle.promise().info;
    }

    Result& get()
    {
      return m_handle.promise().result;
    }

  private:
    handle_type m_handle;
  };
}
