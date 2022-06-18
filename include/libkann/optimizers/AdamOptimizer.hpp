#pragma once

#include <libkann/Export.hpp>

#include <libkann/Optimizer.hpp>

namespace kann
{
  class KANN_EXPORT AdamOptimizer : public Optimizer
  {
  public:
    KANN_EXPORT AdamOptimizer(double alpha, double beta1, double beta2, double epsilon);

  public:
    KANN_EXPORT size_t process(Graph& graph, Info& info, Shape shape, size_t index, size_t gradient_index) const override;

  private:
    double m_alpha;
    double m_beta1, m_beta2;
    double m_epsilon;
  };
}
