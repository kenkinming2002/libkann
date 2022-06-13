#pragma once

#include <libkann/Types.hpp>

#include <vector>
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
}
