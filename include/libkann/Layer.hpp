#pragma once

#include <libkann/Tag.hpp>
#include <libkann/Scope.hpp>
#include <libkann/Types.hpp>

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
    virtual std::vector<QualifiedName> parameters(Scope scope) const { return {}; }
    virtual std::vector<QualifiedName> states(Scope scope) const { return {}; }

    struct Input
    {
      VRef input;
      VMap parameter;
      VMap inputState;
    };

    struct Output
    {
      VRef output;
      VMap outputState;
    };

    virtual Output process(Scope scope, Input input) const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive) {}
  };
}
