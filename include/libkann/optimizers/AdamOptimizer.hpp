#pragma once

#include <libkann/Export.hpp>

#include <libkann/Optimizer.hpp>

namespace kann
{
  class KANN_EXPORT AdamOptimizer : public Optimizer
  {
  public:
    KANN_EXPORT AdamOptimizer(float alpha, float beta1, float beta2, float epsilon);

  public:
    KANN_EXPORT size_t process(Graph& graph, Info& info, Shape shape, size_t index, size_t gradient_index) const override;

  private:
    float m_alpha;
    float m_beta1, m_beta2;
    float m_epsilon;
  };
}
