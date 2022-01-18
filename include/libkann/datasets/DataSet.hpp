#pragma once

#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  class DataSet
  {
  public:
    virtual size_t size() const = 0;

  public:
    virtual std::shared_ptr<const Tensor> get(size_t column, size_t index) const = 0;
    virtual double correctness(size_t column, size_t index, const Tensor& data) const = 0;

  public:
    virtual ~DataSet() = default;
  };
}
