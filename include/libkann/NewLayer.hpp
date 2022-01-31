#pragma once

#include <libkann/NewParameter.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

#include <stddef.h>

namespace kann
{
  enum Tag
  {
    TAG_DEFAULT           = 1u << 0,
    TAG_ENCODDER          = 1u << 1,
    TAG_DECODDER          = 1u << 2,
    TAG_GAN_GENERATOR     = 1u << 3,
    TAG_GAN_DISCRIMINATOR = 1u << 4,
    TAG_ALL = 0xFFFFFFFF
  };

  struct Variable;
  struct LayerVariable
  {
    std::shared_ptr<const Variable> variable;

    std::unordered_map<std::string, std::shared_ptr<const Variable>> parameterVariables;
    std::unordered_map<std::string, std::shared_ptr<const Variable>> stateVariables;
  };

  class NewLayer
  {
  public:
    virtual ~NewLayer() = default;

  public:
    unsigned tag() const { return m_tag; }
    void tag(unsigned tag) { m_tag = tag; }

  public:
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

  public:
    virtual std::vector<NewParameter> parameters() const { return {}; }
    virtual std::vector<NewParameter> stateParameters() const { return {}; }

    virtual LayerVariable operator()(LayerVariable) const = 0;

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
