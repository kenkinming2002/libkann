#pragma once

#include <libkann/Optimizer.hpp>

namespace kann
{
  class SimpleOptimizer : public Optimizer
  {
  public:
    SimpleOptimizer() = default;
    SimpleOptimizer(double learningRate);

  public:
    void process(Context& context) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Optimizer>(this));
      archive(m_learningRate);
    }

  private:
    double m_learningRate;
  };
}

CEREAL_REGISTER_TYPE(kann::SimpleOptimizer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Optimizer, kann::SimpleOptimizer);
