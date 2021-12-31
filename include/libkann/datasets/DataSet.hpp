#pragma once

#include <Eigen/Eigen>

#include <vector>

namespace kann
{
  class DataSet
  {
  public:
    virtual size_t size() const = 0;
    virtual void get(size_t column, size_t index, Eigen::VectorXd& data) const = 0;
    virtual double correctness(size_t column, size_t index, const Eigen::VectorXd& data) const = 0;

  public:
    virtual ~DataSet() = default;
  };
}
