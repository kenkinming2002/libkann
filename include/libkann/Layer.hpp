#pragma once

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
  enum Tag
  {
    TAG_DEFAULT           = 1u << 0,
    TAG_ENCODER           = 1u << 1,
    TAG_DECODER           = 1u << 2,
    TAG_GAN_GENERATOR     = 1u << 3,
    TAG_GAN_DISCRIMINATOR = 1u << 4,
    TAG_ALL = 0xFFFFFFFF
  };

  class Layer
  {
  public:
    virtual ~Layer() = default;

  public:
    unsigned tag() const { return m_tag; }
    void tag(unsigned tag) { m_tag = tag; }

  public:
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

  public:
    virtual std::vector<Parameter> parameters() const { return {}; }
    virtual std::vector<Parameter> stateParameters() const { return {}; }

    virtual LayerVariable operator()(Scope scope, LayerVariable input) const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_tag);
    }

  private:
    unsigned m_tag = TAG_DEFAULT;
  };
}
