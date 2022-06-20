#pragma once

#include <libkann/Export.hpp>
#include <libkann/Tensor.hpp>

namespace kann
{
  struct Function
  {
  public:
    virtual ~Function() = default;

  public:
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor output) = 0;

  public:
    std::vector<Tensor> saved_tensors;
  };
}
