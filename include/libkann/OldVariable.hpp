#pragma once

#include <libkann/layers/Layer.hpp>

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <memory>
#include <vector>
#include <optional>

namespace kann
{
  struct OldVariable
  {
  public:
    static std::shared_ptr<OldVariable> constant(size_t size);

  public:
    struct Input
    {
      std::shared_ptr<OldVariable> variable;
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

  std::shared_ptr<OldVariable> operator|(std::shared_ptr<OldVariable> variable, std::shared_ptr<Layer> layer);
  std::shared_ptr<OldVariable> operator+(std::shared_ptr<OldVariable> lhs, std::shared_ptr<OldVariable> rhs);


}
