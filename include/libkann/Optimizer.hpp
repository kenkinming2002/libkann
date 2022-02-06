#pragma once

#include <libkann/Types.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

namespace kann
{
  class Optimizer
  {
  public:
    virtual ~Optimizer() = default;

  public:
    struct Context
    {
      QualifiedName qualifiedName;

      VRef gradient;

      VRef inputParameter;
      VRef outputParameter;

      VMap inputState;
      VMap outputState;

      TMap initialState;
    };
    virtual void process(Context& context) const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive) {}
  };
}
