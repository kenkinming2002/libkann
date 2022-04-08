#pragma once

#include <libkann/Types.hpp>
#include <libkann/Variable.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Layer.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

namespace kann
{
  class Optimizer
  {
  public:
    virtual ~Optimizer() = default;

  public:
    struct ProcessInput
    {
      size_t size;

      std::shared_ptr<const Variable> parameter;
      std::shared_ptr<const Variable> gradient;
    };

    struct ProcessOutput
    {
      std::shared_ptr<const Variable> parameter;

      std::vector<std::shared_ptr<const Tensor>> initial_states;
      std::vector<std::shared_ptr<const Variable>> input_states;
      std::vector<std::shared_ptr<const Variable>> output_states;
    };

    virtual ProcessOutput process(ProcessInput input) const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive) {}
  };
}
