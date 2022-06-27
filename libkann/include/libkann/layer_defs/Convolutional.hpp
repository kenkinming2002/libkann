#pragma once

#include <libkann/Export.hpp>

#include <libkann/LayerDef.hpp>

#include <libkann/Vec.hpp>

namespace kann
{
  struct KANN_EXPORT ConvolutionalLayerDef : public LayerDef
  {
  public:
    size_t input_channel_count, output_channel_count;
    Vec2 input_size, output_size, kernel_size;

  public:
    KANN_EXPORT static YAML::Node save(std::shared_ptr<const LayerDef> layer_def);
    KANN_EXPORT static std::shared_ptr<const LayerDef> load(YAML::Node node);

  public:
    KANN_EXPORT std::shared_ptr<LayerStorage> create(std::default_random_engine& prng) const override;

  public:
    KANN_EXPORT tensor::Shape get_input_shape() const override;
    KANN_EXPORT tensor::Shape get_output_shape() const override;

  public:
    KANN_EXPORT tensor::Tensor<const float> forward(Layer& layer, tensor::Tensor<const float> inputs) const override;
    KANN_EXPORT tensor::Tensor<const float> backward(Layer& layer, tensor::Tensor<const float> output_gradients) const override;
  };
}
