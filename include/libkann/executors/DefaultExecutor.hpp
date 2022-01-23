#pragma once

#include <libkann/Variable.hpp>

#include <libkann/Executor.hpp>

namespace kann
{
  std::unique_ptr<Executor> makeDefaultExecutor();
}
