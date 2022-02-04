#pragma once

#include <libkann/Types.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

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

  public:
    template<typename Archive>
    void serialize(Archive& archive) {}
  };
}
