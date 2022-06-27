#pragma once

#include <libkann/Function.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/LayerStorage.hpp>

namespace kann
{
  struct KANN_EXPORT Layer : public Function
  {
  public:
    KANN_EXPORT static std::shared_ptr<Layer> create_from(std::shared_ptr<const LayerDef> def, std::shared_ptr<LayerStorage> storage);

  public:
    std::shared_ptr<const LayerDef> def;
    std::shared_ptr<LayerStorage> storage;

  public:
    std::vector<std::shared_ptr<Layer>> sub_layers;

  public:
    KANN_EXPORT Tensor<const float> forward(Tensor<const float> inputs) override;
    KANN_EXPORT Tensor<const float> backward(Tensor<const float> outputs) override;
  };
}
