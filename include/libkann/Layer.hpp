#pragma once

#include <libkann/Tag.hpp>
#include <libkann/LayerVariable.hpp>
#include <libkann/Scope.hpp>

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
    Tag tag() const { return m_tag; }
    void tag(Tag tag) { m_tag = tag; }

  public:
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

  public:
    virtual std::vector<Parameter> parameters(Scope scope) const { return {}; }
    virtual std::vector<Parameter> stateParameters(Scope scope) const { return {}; }

    virtual LayerVariable operator()(Scope scope, LayerVariable input) const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_tag);
    }

  private:
    Tag m_tag = Tag::ALL;
  };
}
