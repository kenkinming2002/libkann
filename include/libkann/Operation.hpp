#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Variable.hpp>

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
    virtual CRef<Tensor> process(std::vector<const Tensor*> inputs) const = 0;
    virtual std::vector<CRef<Variable>> gradients(CRef<Variable> gradient, std::vector<CRef<Variable>> inputs) const = 0;
  };

  class UnaryOperation : public Operation
  {
  public:
    CRef<Tensor> process(std::vector<const Tensor*> inputs) const override final
    {
      assert(inputs.size() == 1);
      return std::make_shared<const Tensor>(this->processImpl(*inputs[0]));
    }

    std::vector<CRef<Variable>> gradients(CRef<Variable> gradient, std::vector<CRef<Variable>> inputs) const override
    {
      assert(inputs.size() == 1);
      auto result = this->gradientsImpl(std::move(gradient), std::move(inputs[0]));
      return {std::move(result)};
    }

  protected:
    virtual Tensor processImpl(const Tensor&) const = 0;
    virtual CRef<Variable> gradientsImpl(CRef<Variable> gradient, CRef<Variable>) const = 0;
  };

  class BinaryOperation : public Operation
  {
  public:
    CRef<Tensor> process(std::vector<const Tensor*> inputs) const override final
    {
      assert(inputs.size() == 2);
      return std::make_shared<const Tensor>(this->processImpl(*inputs[0], *inputs[1]));
    }

    std::vector<CRef<Variable>> gradients(CRef<Variable> gradient, std::vector<CRef<Variable>> inputs) const override
    {
      assert(inputs.size() == 2);
      auto result = this->gradientsImpl(std::move(gradient), std::move(inputs[0]), std::move(inputs[1]));
      return {std::move(result.first), std::move(result.second)};
    }

  protected:
    virtual Tensor processImpl(const Tensor&, const Tensor&) const = 0;
    virtual std::pair<CRef<Variable>, CRef<Variable>> gradientsImpl(CRef<Variable> gradient, CRef<Variable>, CRef<Variable>) const = 0;
  };
}
