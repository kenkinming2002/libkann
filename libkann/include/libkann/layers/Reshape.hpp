#pragma once

#include <libkann/Export.hpp>

#include <libkann/LayerDef.hpp>
#include <libkann/Layer.hpp>

namespace kann
{
  struct LIBKANN_EXPORT ReshapeLayerDef final : public LayerDef
  {
  public:
    tensor::Shape input_shape;
    tensor::Shape output_shape;

  public:
    LIBKANN_EXPORT std::unique_ptr<Layer> create() const override;
  };

  struct LIBKANN_EXPORT ReshapeLayer final : public Layer
  {
  public:
    ReshapeLayerDef def;

  public:
    LIBKANN_EXPORT const LayerDef& get_def() const override;

    LIBKANN_EXPORT void initialize(std::default_random_engine& prng) override;

    LIBKANN_EXPORT std::unordered_map<std::string, const Variable*> parameters_map() const override;
    LIBKANN_EXPORT std::unordered_map<std::string, Variable*>       parameters_map()       override;

    LIBKANN_EXPORT tensor::Tensor<float> forward(tensor::Tensor<float> inputs) override;
    LIBKANN_EXPORT tensor::Tensor<float> backward(tensor::Tensor<float> output_gradients) override;
  };
}

