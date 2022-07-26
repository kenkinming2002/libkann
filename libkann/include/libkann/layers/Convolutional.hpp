#pragma once

#include <libkann/Export.hpp>

#include <libkann/LayerDef.hpp>
#include <libkann/Layer.hpp>

#include <libtensor/Vec.hpp>

namespace kann
{
  struct LIBKANN_EXPORT ConvolutionalLayerDef : public LayerDef
  {
  public:
    size_t input_channel_count, output_channel_count;
    tensor::Vec2 input_size, output_size, kernel_size;

  public:
    LIBKANN_EXPORT std::unique_ptr<Layer> create() const override;
  };

  struct ConvolutionalLayer : public Layer
  {
  public:
    ConvolutionalLayerDef def;
    Variable kernels;

  public:
    LIBKANN_EXPORT const LayerDef& get_def() const override;

    LIBKANN_EXPORT void initialize(std::default_random_engine& prng) override;

    LIBKANN_EXPORT std::unordered_map<std::string, const Variable*> parameters_map() const override;
    LIBKANN_EXPORT std::unordered_map<std::string, Variable*>       parameters_map()       override;

    LIBKANN_EXPORT tensor::Tensor<float> forward(tensor::Tensor<float> inputs) override;
    LIBKANN_EXPORT tensor::Tensor<float> backward(tensor::Tensor<float> output_gradients) override;
  };
}
