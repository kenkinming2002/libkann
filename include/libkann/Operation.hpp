#pragma once

#include <functional>
#include <libkann/Tensor.hpp>

#include <vector>
#include <utility>
#include <memory>

namespace kann
{
  class Variable;

  typedef std::shared_ptr<const Variable> VariableHandle;
  typedef std::pair<VariableHandle, VariableHandle> VariablePair;
  typedef std::vector<VariableHandle> VariableList;

  class Operation
  {
  public:
    virtual ~Operation() = default;

  public:
    virtual Tensor process(std::vector<std::reference_wrapper<const Tensor>> inputs) const = 0;
    virtual VariableList gradients(VariableHandle gradient, VariableList inputs) const = 0;
  };

  class UnaryOperation : public Operation
  {
  public:
    Tensor process(std::vector<std::reference_wrapper<const Tensor>> inputs) const override final
    {
      assert(inputs.size() == 1);
      return this->processImpl(inputs[0]);
    }

    VariableList gradients(VariableHandle gradient, VariableList inputs) const override
    {
      assert(inputs.size() == 1);
      auto result = this->gradientsImpl(std::move(gradient), std::move(inputs[0]));
      return {std::move(result)};
    }

  protected:
    virtual Tensor processImpl(const Tensor&) const = 0;
    virtual VariableHandle gradientsImpl(VariableHandle gradient, VariableHandle) const = 0;
  };

  class BinaryOperation : public Operation
  {
  public:
    Tensor process(std::vector<std::reference_wrapper<const Tensor>> inputs) const override final
    {
      assert(inputs.size() == 2);
      return this->processImpl(inputs[0], inputs[1]);
    }

    VariableList gradients(VariableHandle gradient, VariableList inputs) const override
    {
      assert(inputs.size() == 2);
      auto result = this->gradientsImpl(std::move(gradient), std::move(inputs[0]), std::move(inputs[1]));
      return {std::move(result.first), std::move(result.second)};
    }

  protected:
    virtual Tensor processImpl(const Tensor&, const Tensor&) const = 0;
    virtual VariablePair gradientsImpl(VariableHandle gradient, VariableHandle, VariableHandle) const = 0;
  };
}
