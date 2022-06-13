#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libkann/Graph.hpp>

#include <libkann/operations/Scale.hpp>
#include <libkann/operations/Subtract.hpp>

namespace kann
{
  SimpleOptimizer::SimpleOptimizer(double learningRate)
    : m_learningRate(learningRate) {}

  size_t SimpleOptimizer::process(Graph& graph, Info& info, size_t size, size_t index, size_t gradient_index) const
  {
    size_t new_index = graph.add_vertex();
    size_t tmp_index = graph.add_vertex();

    operation_t scale_op    = std::make_shared<ScaleOperation>(size, m_learningRate);
    operation_t subtract_op = std::make_shared<SubtractOperation>(size);
    graph.add_edge(scale_op, {gradient_index}, {tmp_index});
    graph.add_edge(subtract_op, {index, tmp_index}, {new_index});

    return new_index;
  }
}
