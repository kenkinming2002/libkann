#pragma once

#include <libkann/Tag.hpp>
#include <libkann/Scope.hpp>
#include <libkann/Types.hpp>
#include <libkann/Variable.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

#include <memory>
#include <vector>

#include <assert.h>
#include <stddef.h>

namespace kann
{
  class Layer
  {
  public:
    virtual ~Layer() = default;

  public:
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

  public:
    struct Parameter
    {
      QualifiedName name;

      size_t size;

      // TODO: Abstract initializer base class
      double mean;
      double stddev;
    };

    struct State
    {
      QualifiedName name;

      size_t size;

      // Question: Do we want random state initialization or only allow zero
      //           initialization?
    };

    virtual std::vector<Parameter> parameters(Scope scope) const { return {}; }
    virtual std::vector<State> states(Scope scope) const { return {}; }

    struct Input
    {
      CRef<Variable> input;
      Map<Variable> parameter;
      Map<Variable> inputState;
    };

    struct Output
    {
      CRef<Variable> output;
      Map<Variable> outputState;
    };

    virtual Output process(Scope scope, Input input) const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive) {}
  };
}
