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
    void process(Context& context) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Optimizer>(this));
      archive(m_alpha);
    }

  private:
    double m_alpha;
    double m_beta1, m_beta2;
    double m_epsilon;
  };
}

CEREAL_REGISTER_TYPE(kann::AdamOptimizer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Optimizer, kann::AdamOptimizer);

