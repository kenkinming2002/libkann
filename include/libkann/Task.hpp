#pragma once

#include <coroutine>

namespace kann
{
  namespace coroutine
  {
    template<typename T>
    struct PromiseYield
    {
    public:
      auto yield_value(T t)
      {
        m_t = std::move(t);
        return std::suspend_always{};
      }

      T getYield()
      {
        return std::move(m_t);
      }

    private:
      T m_t;
    };

    template<>
    struct PromiseYield<void>
    {
    public:
      void getYield() {}
    };

    template<typename T>
    struct PromiseReturn
    {
    public:
      void return_value(T t)
      {
        m_t = std::move(t);
      }

      T getReturn()
      {
        return std::move(m_t);
      }

    private:
      T m_t;
    };

    template<>
    struct PromiseReturn<void>
    {
    public:
      void return_void() {}
      void getReturn() {}
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

      auto initial_suspend() noexcept { return std::suspend_always{}; }
      auto final_suspend()   noexcept { return std::suspend_always{}; }

      void unhandled_exception() {}
    };

  public:
    Task(std::coroutine_handle<promise_type> handle) : m_handle(handle) {}
    ~Task() { if(m_handle) m_handle.destroy(); }

  public:
    bool step()
    {
      m_handle.resume();
      return m_handle.done();
    }

    decltype(auto) info() { return m_handle.promise().getYield(); }
    decltype(auto) get()  { return m_handle.promise().getReturn(); }

  private:
    std::coroutine_handle<promise_type> m_handle;
  };

}
