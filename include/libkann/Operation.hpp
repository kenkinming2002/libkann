#pragma once

#include <libkann/Utility.hpp>
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
    virtual std::vector<tensor_t> process(std::vector<tensor_t> inputs) const
    {
      assert(false && "Unimplemented");
    }

    /* Given an M to N operation op,
     * op.differentiate() is an M+N to M operation
     * where the M+N inputs are:
     *
     * 1: M original inputs
     * 2: N output gradients */
    virtual operation_t differentiate() const
    {
      assert(false && "Unimplemented");
    }
  };

  template<typename Derived, size_t M, size_t N>
  class OperationImpl : public Operation
  {
  public:
    using inputs_t  = std::array<tensor_t, M>;
    using outputs_t = std::array<Tensor, N>;

  public:
    outputs_t process_impl(inputs_t inputs) const = delete;

  private:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      inputs_t _inputs = inputs | to_array<tensor_t, M>();
      outputs_t _outputs = derived().process_impl(std::move(_inputs));
      return _outputs
        | ranges::views::transform([](Tensor& v) -> tensor_t { return std::make_shared<const Tensor>(std::move(v)); })
        | ranges::to_vector;
    }
  };
}
