#pragma once

#include <libkann/Tag.hpp>
#include <libkann/Types.hpp>
#include <libkann/Variable.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

#include <memory>
#include <vector>
#include <random>

#include <assert.h>
#include <stddef.h>

namespace kann
{
  class Layer
  {
  protected:
    static std::shared_ptr<const Tensor> create_tensor_gaussian(size_t size, double mean, double variance, std::default_random_engine& engine)
    {
      std::normal_distribution dist(mean, variance);
      return std::make_shared<const Tensor>(Tensor::nullaryExpr(size, [&](){ return dist(engine); }));
    }

  public:
    virtual ~Layer() = default;

  public:
    virtual std::shared_ptr<Layer> clone() const = 0;
    virtual void randomize(std::default_random_engine& /*engine*/) {}

  public:
    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;

  public:
    virtual size_t parameters_count() const { return 0; }
    virtual size_t states_count() const { return 0; }

    virtual std::vector<size_t> parameter_sizes() const { return {}; }
    virtual std::vector<size_t> state_sizes() const { return {}; }

    virtual std::vector<std::shared_ptr<const Tensor>> get_parameters() const { return {}; }
    virtual std::vector<std::shared_ptr<const Tensor>> get_states() const { return {}; }

    virtual void set_parameters(std::vector<std::shared_ptr<const Tensor>> values) { assert(values.empty()); }
    virtual void set_states(std::vector<std::shared_ptr<const Tensor>> values) { assert(values.empty()); }

  public:
    struct ProcessInput
    {
      std::shared_ptr<const Variable> variable;
      std::vector<std::shared_ptr<const Variable>> parameters;
      std::vector<std::shared_ptr<const Variable>> states;
    };

    struct ProcessOutput
    {
      std::shared_ptr<const Variable> variable;
      std::vector<std::shared_ptr<const Variable>> states;
    };

    virtual ProcessOutput process(ProcessInput input) const = 0;

  public:
    template<typename Archive> void serialize(Archive& archive) {}
  };
}
