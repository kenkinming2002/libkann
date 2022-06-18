#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <vector>
#include <assert.h>

namespace kann
{
  class Operation
  {
  public:
    KANN_EXPORT virtual ~Operation() = default;

  public:
    KANN_EXPORT virtual std::vector<Tensor> process(std::vector<Tensor> inputs) const
    {
      assert(false && "Unimplemented");
    }

    /* Given an M to N operation op,
     * op.differentiate() is an M+N to M operation
     * where the M+N inputs are:
     *
     * 1: M original inputs
     * 2: N output gradients */
    KANN_EXPORT virtual operation_t differentiate() const
    {
      assert(false && "Unimplemented");
    }
  };
}
