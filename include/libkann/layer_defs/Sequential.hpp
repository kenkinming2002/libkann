#pragma once

#include <libkann/Export.hpp>

#include <libkann/LayerDef.hpp>

namespace kann
{
  class KANN_EXPORT SequentialLayerDef : public LayerDef
  {
  public:
    KANN_EXPORT static YAML::Node save(std::shared_ptr<const LayerDef> layer_def);
    KANN_EXPORT static std::shared_ptr<const LayerDef> load(YAML::Node node);

  public:
    KANN_EXPORT std::shared_ptr<LayerStorage> create(std::default_random_engine& prng) const override;

  public:
    KANN_EXPORT Shape input_shape() const override;
    KANN_EXPORT Shape output_shape() const override;

  public:
    KANN_EXPORT Tensor<float> forward(Layer& layer, Tensor<float> inputs) const override;
    KANN_EXPORT Tensor<float> backward(Layer& layer, Tensor<float> output_gradients) const override;
  };
}
