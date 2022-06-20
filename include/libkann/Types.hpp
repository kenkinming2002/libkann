#pragma once

#include <memory>

namespace kann
{
  class Operation;

  struct LayerDef;
  struct LayerStorage;

  class Optimizer;

  class Graph;
  class Executor;

  typedef std::shared_ptr<const Operation> operation_t;

  typedef std::shared_ptr<const LayerDef> std::shared_ptr<const LayerDef>;
  typedef std::shared_ptr<LayerStorage> layer_t;

  typedef std::shared_ptr<const Optimizer> optimizer_t;

  typedef std::shared_ptr<const Graph> graph_t;
}
