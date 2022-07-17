#pragma once

#include <libkann/Function.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/LayerStorage.hpp>

namespace kann
{
  struct KANN_EXPORT Layer : public Function
  {
  public:
    KANN_EXPORT static std::shared_ptr<Layer> from(std::shared_ptr<const LayerDef> def, std::shared_ptr<LayerStorage> storage);
    KANN_EXPORT static std::shared_ptr<Layer> load_and_create_from(const std::string& filename, std::default_random_engine& prng);

  public:
    std::shared_ptr<const LayerDef> def;
    std::shared_ptr<LayerStorage> storage;

  public:
    std::vector<std::shared_ptr<Layer>> sub_layers;

  public:
    KANN_EXPORT tensor::Tensor<float> forward(tensor::Tensor<float> inputs) override;
    KANN_EXPORT tensor::Tensor<float> backward(tensor::Tensor<float> outputs) override;
  };
}
