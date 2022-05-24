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
    ProcessOutput process(ProcessInput input) const override;
    std::vector<std::shared_ptr<const Tensor>> create_initial_states(size_t size) const override;

  private:
    double m_alpha;
    double m_beta1, m_beta2;
    double m_epsilon;
  };
}
