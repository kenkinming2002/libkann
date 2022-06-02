#pragma once

#include <coroutine>
#include <optional>
#include <memory>
#include <exception>

namespace kann
{
  // Implemention of coroutine task type
  // with progress report
  namespace coroutine
  {
    template<typename T>
    struct PromiseYield
    {
    public:
      auto yield_value(T& t)
      {
        m_t = std::addressof(t);
        return std::suspend_always{};
      }

      T& get_yield()
      {
        return *m_t;
      }

    private:
      T* m_t;
    };

    template<>
    struct PromiseYield<void>
    {
    public:
      void get_yield() {}
    };

    template<typename T>
    struct PromiseReturn
    {
    public:
      void return_value(T t)
      {
        m_t.emplace(std::move(t));
      }

      T& get_return()
      {
        return *m_t;
      }

    private:
      std::optional<T> m_t;
    };

    template<>
    struct PromiseReturn<void>
    {
    public:
      void return_void() {}
      void get_return() {}
    };
  }

  template<typename Result, typename Info>
  struct Task
  {
  public:
    struct promise_type :
      public coroutine::PromiseReturn<Result>,
      public coroutine::PromiseYield<Info>
    {
    public:
      Task get_return_object()
      {
        auto handle = std::coroutine_handle<promise_type>::from_promise(*this);
        return Task(handle);
      }

      auto initial_suspend() noexcept { return std::suspend_never{}; }
      auto final_suspend()   noexcept { return std::suspend_always{}; }

      void unhandled_exception()
      {
        std::terminate();
      }
    };

  public:
    Task(std::coroutine_handle<promise_type> handle) : m_handle(handle) {}
    ~Task() { if(m_handle) m_handle.destroy(); }

  public:
    void step()
    {
      m_handle.resume();
    }

    bool is_done() const
    {
      return m_handle.done();
    }

    decltype(auto) info() { return m_handle.promise().get_yield(); }
    decltype(auto) get()  { return m_handle.promise().get_return(); }

  private:
    std::coroutine_handle<promise_type> m_handle;
  };

}
