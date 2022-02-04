#pragma once

#include <libkann/Types.hpp>

namespace kann
{
  class Optimizer
  {
  public:
    virtual ~Optimizer() = default;

  public:
    /* Given variable for parameter and gradient, return variable for new
     * parameter */
    virtual std::vector<Parameter> stateParameters() { return  {}; }

    struct Result
    {
      VRef newParameter;
      VMap newState;
    };
    virtual Result process(VRef parameter, VRef gradient, VMap state) const = 0;
  };
}
