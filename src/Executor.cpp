#include <libkann/Executor.hpp>

#include <libkann/executors/DefaultExecutor.hpp>

namespace kann
{
  std::unique_ptr<Executor> makeDefaultExecutor()
  {
    return std::make_unique<DefaultExecutor>();
  }
}
