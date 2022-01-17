#pragma once

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <vector>

namespace kann
{
  struct Tensor
  {
  public:
    Tensor() = default;
    Tensor(size_t size) : values(size) {}
    Tensor(Eigen::VectorXd data);

  public:
    std::vector<double> values;

  public:
    size_t size() const { return values.size(); }

  public:
    auto asArray() &
    {
      return Eigen::ArrayXd::Map(values.data(), values.size());
    }

    auto asArray() const &
    {
      return Eigen::ArrayXd::Map(values.data(), values.size());
    }

  public:
    auto asVector() &
    {
      return Eigen::VectorXd::Map(values.data(), values.size());
    }

    auto asVector() const &
    {
      return Eigen::VectorXd::Map(values.data(), values.size());
    }

  public:
    auto asMatrix(size_t rows, size_t cols) &
    {
      assert(values.size() == rows * cols);
      return Eigen::MatrixXd::Map(values.data(), rows, cols);
    }

    auto asMatrix(size_t rows, size_t cols) const &
    {
      assert(values.size() == rows * cols);
      return Eigen::MatrixXd::Map(values.data(), rows, cols);
    }

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(values);
    }
  };

  inline Tensor::Tensor(Eigen::VectorXd v) : Tensor(v.size())
  {
    asVector() = v;
  }
}
