#pragma once

#include <libkann/Export.hpp>

#include <libkann/LayerDef.hpp>

namespace kann
{
  struct KANN_EXPORT SoftMaxLayerDef : public LayerDef
  {
  public:
    tensor::Shape shape;

  public:
    KANN_EXPORT static YAML::Node save(std::shared_ptr<const LayerDef> layer_def);
    KANN_EXPORT static std::shared_ptr<const LayerDef> load(YAML::Node node);

  public:
    KANN_EXPORT std::shared_ptr<LayerStorage> create(std::default_random_engine& prng) const override;

  public:
    KANN_EXPORT tensor::Shape get_input_shape() const override;
    KANN_EXPORT tensor::Shape get_output_shape() const override;

  public:
    KANN_EXPORT tensor::Tensor<float> forward(Layer& layer, tensor::Tensor<float> inputs) const override;
    KANN_EXPORT tensor::Tensor<float> backward(Layer& layer, tensor::Tensor<float> output_gradients) const override;
  };
}
