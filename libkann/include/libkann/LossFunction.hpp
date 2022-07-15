#pragma once

#include <libkann/Export.hpp>
#include <libkann/Function.hpp>

#include <optional>

namespace kann
{
  struct KANN_EXPORT LossFunction : public Function
  {
  public:
    tensor::Shape shape;
    std::optional<tensor::Tensor<const float>> expected_outputs;
  };
}
