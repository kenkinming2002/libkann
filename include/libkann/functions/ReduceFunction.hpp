#pragma once

#include <libkann/Function.hpp>

namespace kann
{
  class ReduceFunction : public UnaryFunction
  {
  public:
    ReduceFunction(size_t inputCount);

  protected:
    std::shared_ptr<const Variable> impl(std::shared_ptr<const Variable>) const override;

  private:
    size_t m_inputCount;
  };
}

