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
    KANN_EXPORT void step() override;
    KANN_EXPORT void optimize(Variable& variable) const override;

  private:
    float m_alpha;
    float m_beta1, m_beta2;
    float m_epsilon;

  private:
    size_t m_timestep = 1;
  };
}
