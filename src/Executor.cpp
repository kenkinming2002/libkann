#include <libkann/Executor.hpp>

#include <libkann/executors/DefaultExecutor.hpp>
#include <libkann/executors/ThreadedExecutor.hpp>

namespace kann
{
  std::unique_ptr<Executor> makeDefaultExecutor()
  {
    return std::make_unique<DefaultExecutor>();
  }

  std::unique_ptr<Executor> makeThreadedExecutor()
  {
    return std::make_unique<ThreadedExecutor>();
  }
}
