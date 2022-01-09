#pragma once

#include <libkann/layers/Layer.hpp>

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <memory>
#include <vector>
#include <optional>

namespace kann
{
  struct Variable
  {
  public:
    static std::shared_ptr<Variable> constant(size_t size);

  public:
    struct Input
    {
      std::shared_ptr<Variable> variable;
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

  public:
    std::vector<Input> inputs;

  public:
    Eigen::VectorXd data;
    Eigen::RowVectorXd gradient;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(inputs);
      archive(data);
      archive(gradient);
    }
  };

  std::shared_ptr<Variable> operator|(std::shared_ptr<Variable> variable, std::shared_ptr<Layer> layer);
  std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs);


}
