#pragma once

#include <libkann/Tag.hpp>
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
  class Layer : public std::enable_shared_from_this<Layer>
  {
  public:
    virtual ~Layer() = default;

  public:
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

  public:
    struct Parameter
    {
      std::shared_ptr<const Layer> layer;
      std::string name;

      size_t size;
      double mean, stddev;
      template<typename Archive> void serialize(Archive& archive) { archive(layer, name, size, mean, stddev); }
    };

    struct State
    {
      std::shared_ptr<const Layer> layer;
      std::string name;

      size_t size;
      template<typename Archive> void serialize(Archive& archive) { archive(layer, name, size); }
    };

  public:
    virtual std::vector<Parameter> parameters() const { return {}; };
    virtual std::vector<State> states() const { return {}; };

  public:
    struct ProcessInput
    {
      std::shared_ptr<const Variable> variable;
      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> parameters;
      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> states;
    };

    struct ProcessOutput
    {
      std::shared_ptr<const Variable> variable;
      std::map<std::pair<std::shared_ptr<const Layer>, std::string>, std::shared_ptr<const Variable>> states;
    };

  public:
    virtual ProcessOutput process(ProcessInput input) const = 0;

  public:
    template<typename Archive> void serialize(Archive& archive) {}
  };
}
