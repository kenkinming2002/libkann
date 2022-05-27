#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <vector>
#include <utility>

#include <assert.h>

namespace kann
{
  class Operation
  {
  public:
    virtual ~Operation() = default;

  public:
    virtual tensor_t process(std::vector<const Tensor*> inputs) const = 0;
    virtual std::vector<variable_t> gradients(variable_t gradient, std::vector<variable_t> inputs) const = 0;
  };

  class UnaryOperation : public Operation
  {
  public:
    tensor_t process(std::vector<const Tensor*> inputs) const override final
    {
      assert(inputs.size() == 1);
      return std::make_shared<const Tensor>(this->processImpl(*inputs[0]));
    }

    std::vector<variable_t> gradients(variable_t gradient, std::vector<variable_t> inputs) const override
    {
      assert(inputs.size() == 1);
      auto result = this->gradientsImpl(std::move(gradient), std::move(inputs[0]));
      return {std::move(result)};
    }

  protected:
    virtual Tensor processImpl(const Tensor&) const = 0;
    virtual variable_t gradientsImpl(variable_t gradient, variable_t) const = 0;
  };

  class BinaryOperation : public Operation
  {
  public:
    tensor_t process(std::vector<const Tensor*> inputs) const override final
    {
      assert(inputs.size() == 2);
      return std::make_shared<const Tensor>(this->processImpl(*inputs[0], *inputs[1]));
    }

    std::vector<variable_t> gradients(variable_t gradient, std::vector<variable_t> inputs) const override
    {
      assert(inputs.size() == 2);
      auto result = this->gradientsImpl(std::move(gradient), std::move(inputs[0]), std::move(inputs[1]));
      return {std::move(result.first), std::move(result.second)};
    }

  protected:
    virtual Tensor processImpl(const Tensor&, const Tensor&) const = 0;
    virtual std::pair<variable_t, variable_t> gradientsImpl(variable_t gradient, variable_t, variable_t) const = 0;
  };
}
