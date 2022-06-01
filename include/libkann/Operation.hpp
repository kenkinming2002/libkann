#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <range/v3/all.hpp>

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

  template<typename Derived, size_t N>
  class OperationImpl : public Operation
  {
  public:
    using inputs_t    = std::array<const Tensor*, N>;
    using variables_t = std::array<variable_t, N>;

  public:
    Tensor process_impl(inputs_t inputs) const = delete;
    variables_t gradients_impl(variable_t gradient, variables_t inputs) const = delete;

  public:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

  public:
    tensor_t process(std::vector<const Tensor*> inputs) const override
    {
      inputs_t _inputs;
      assert(inputs.size() == N);
      ranges::move(inputs, _inputs.begin());
      return std::make_shared<const Tensor>(derived().process_impl(_inputs));
    }

    std::vector<variable_t> gradients(variable_t gradient, std::vector<variable_t> inputs) const override
    {
      variables_t _inputs;
      assert(inputs.size() == N);
      ranges::move(inputs, _inputs.begin());
      return derived().gradients_impl(gradient, _inputs) | ranges::to_vector;
    }
  };
}
