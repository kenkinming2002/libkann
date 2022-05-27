#pragma once

#include <memory>

namespace kann
{
  struct Tensor;

  class Variable;
  class Operation;

  struct LayerDef;
  struct Layer;

  class Optimizer;

  class Graph;
  class Executor;

  typedef std::shared_ptr<const Tensor> tensor_t;

  typedef std::shared_ptr<const Operation> operation_t;
  typedef std::shared_ptr<const Variable> variable_t;

  typedef std::shared_ptr<const LayerDef> layer_def_t;
  typedef std::shared_ptr<Layer> layer_t;

  typedef std::shared_ptr<const Optimizer> optimizer_t;

  typedef std::shared_ptr<const Graph> graph_t;
}
