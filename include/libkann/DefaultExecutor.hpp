#pragma once

#include <libkann/Variable.hpp>

#include <libkann/Executor.hpp>

namespace kann
{
  std::unique_ptr<Executor> makeDefaultExecutor(std::vector<std::shared_ptr<const Variable>> inputs, std::vector<std::shared_ptr<const Variable>> outputs);
}
