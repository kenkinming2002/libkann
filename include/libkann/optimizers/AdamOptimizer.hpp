#pragma once

#include <libkann/Optimizer.hpp>

namespace kann
{
  class AdamOptimizer : public Optimizer
  {
  public:
    AdamOptimizer() = default;
    AdamOptimizer(double alpha, double beta1, double beta2, double epsilon);

  public:
    size_t process(Graph& graph, Info& info, Shape shape, size_t index, size_t gradient_index) const override;

  private:
    double m_alpha;
    double m_beta1, m_beta2;
    double m_epsilon;
  };
}
