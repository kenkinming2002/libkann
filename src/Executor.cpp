#include <libkann/Executor.hpp>

#include <libkann/executors/DefaultExecutor.hpp>
//#include <libkann/executors/ThreadedExecutor.hpp>

#include <assert.h>

namespace kann
{
  std::unique_ptr<Executor> Executor::create(Type type)
  {
    switch(type)
    {
    case Type::DEFAULT:
      return std::make_unique<DefaultExecutor>();
    case Type::THREADED:
      //return std::make_unique<ThreadedExecutor>();
    default:
      assert(false);
    }
  }
}
