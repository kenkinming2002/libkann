#pragma once

#include <Eigen/Eigen>

#include <vector>

namespace kann
{
  struct Tensor
  {
  public:
    Tensor() = default;
    Tensor(size_t size) : values(size) {}

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
  };
}
