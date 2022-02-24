#pragma once

#include <libkann/Types.hpp>

#include <functional>
#include <vector>
#include <utility>
#include <memory>

namespace kann
{
  class Operation
  {
  public:
    virtual ~Operation() = default;

  public:
    virtual TRef process(std::vector<const Tensor*> inputs) const = 0;
    virtual std::vector<VRef> gradients(VRef gradient, std::vector<VRef> inputs) const = 0;
  };

  class UnaryOperation : public Operation
  {
  public:
    TRef process(std::vector<const Tensor*> inputs) const override final
    {
      assert(inputs.size() == 1);
      return std::make_shared<const Tensor>(this->processImpl(*inputs[0]));
    }

    std::vector<VRef> gradients(VRef gradient, std::vector<VRef> inputs) const override
    {
      assert(inputs.size() == 1);
      auto result = this->gradientsImpl(std::move(gradient), std::move(inputs[0]));
      return {std::move(result)};
    }

  protected:
    virtual Tensor processImpl(const Tensor&) const = 0;
    virtual VRef gradientsImpl(VRef gradient, VRef) const = 0;
  };

  class BinaryOperation : public Operation
  {
  public:
    TRef process(std::vector<const Tensor*> inputs) const override final
    {
      assert(inputs.size() == 2);
      return std::make_shared<const Tensor>(this->processImpl(*inputs[0], *inputs[1]));
    }

    std::vector<VRef> gradients(VRef gradient, std::vector<VRef> inputs) const override
    {
      assert(inputs.size() == 2);
      auto result = this->gradientsImpl(std::move(gradient), std::move(inputs[0]), std::move(inputs[1]));
      return {std::move(result.first), std::move(result.second)};
    }

  protected:
    virtual Tensor processImpl(const Tensor&, const Tensor&) const = 0;
    virtual std::pair<VRef, VRef> gradientsImpl(VRef gradient, VRef, VRef) const = 0;
  };
}
