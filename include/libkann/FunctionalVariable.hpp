#pragma once

#include <libkann/layers/Layer.hpp>

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <memory>
#include <vector>
#include <optional>

namespace kann
{
  struct FunctionalVariable
  {
  public:
    static std::shared_ptr<FunctionalVariable> constant(size_t size);

  public:
    struct Input
    {
      std::shared_ptr<FunctionalVariable> variable;
      std::shared_ptr<Layer> layer;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(variable);
        archive(layer);
      }
    };

  public:
    size_t size;
    std::vector<Input> inputs;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(inputs);
    }
  };

  std::shared_ptr<FunctionalVariable> operator|(std::shared_ptr<FunctionalVariable> variable, std::shared_ptr<Layer> layer);
  std::shared_ptr<FunctionalVariable> operator+(std::shared_ptr<FunctionalVariable> lhs, std::shared_ptr<FunctionalVariable> rhs);


}
