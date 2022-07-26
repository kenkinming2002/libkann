#pragma once

#include <libkann/Export.hpp>
#include <libkann/Function.hpp>

#include <optional>

namespace kann
{
  struct LIBKANN_EXPORT LossFunction : public Function
  {
  public:
    std::optional<tensor::Tensor<float>> expected_outputs;
  };
}
