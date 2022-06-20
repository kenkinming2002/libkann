#pragma once

#include <libkann/Function.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/LayerStorage.hpp>

namespace kann
{
  struct KANN_EXPORT Layer : public Function
  {
  public:
    std::shared_ptr<const LayerDef> def;
    std::shared_ptr<LayerStorage> storage;

  public:
    KANN_EXPORT Tensor forward(Tensor inputs) override;
    KANN_EXPORT Tensor backward(Tensor outputs) override;
  };
}
