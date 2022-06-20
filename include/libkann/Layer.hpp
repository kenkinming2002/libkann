#pragma once

#include <libkann/Function.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/LayerStorage.hpp>

namespace kann
{
  struct Layer : public Function
  {
  public:
    std::shared_ptr<const LayerDef> def;
    std::shared_ptr<LayerStorage> storage;

  public:
    Tensor forward(Tensor inputs) override;
    Tensor backward(Tensor outputs) override;
  };
}
