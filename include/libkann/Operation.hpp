#pragma once

#include <libkann/Tensor.hpp>

#include <vector>
#include <memory>

namespace kann
{
  class Function;

  class Operation
  {
  public:
    virtual ~Operation() = default;

  public:
    virtual Tensor process(const std::vector<Tensor>& inputs) const = 0;
    virtual std::shared_ptr<const Function> function() const = 0;
  };

  class UnaryOperation : public Operation
  {
  public:
    Tensor process(const std::vector<Tensor>& inputs) const override final
    {
      assert(inputs.size() == 1);
      return this->processImpl(inputs[0]);
    }

  protected:
    virtual Tensor processImpl(const Tensor&) const = 0;
  };

  class BinaryOperation : public Operation
  {
  public:
    Tensor process(const std::vector<Tensor>& inputs) const override final
    {
      assert(inputs.size() == 2);
      return this->processImpl(inputs[0], inputs[1]);
    }

  protected:
    virtual Tensor processImpl(const Tensor&, const Tensor&) const = 0;
  };
}
