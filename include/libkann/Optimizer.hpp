#pragma once

#include <libkann/Types.hpp>
#include <libkann/Variable.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Layer.hpp>

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
      Layer::Parameter parameter;

      CRef<Variable> gradient;

      CRef<Variable> inputParameter;
      CRef<Variable> outputParameter;

      Map<Variable> inputState;
      Map<Variable> outputState;

      Map<Tensor> initialState;
    };
    virtual void process(Context& context) const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive) {}
  };
}
