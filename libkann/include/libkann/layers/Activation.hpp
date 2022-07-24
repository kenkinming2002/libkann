#pragma once

#include <libkann/Export.hpp>

#include <libkann/LayerDef.hpp>
#include <libkann/Layer.hpp>

namespace kann
{
  struct KANN_EXPORT ActivationLayerDef final : public LayerDef
  {
  public:
    enum class Type
    {
      IDENTITY,
      SIGMOID,
      TANH
    };

  public:
    tensor::Shape shape;
    Type type;

  public:
    KANN_EXPORT std::shared_ptr<Layer> create() const override;
  };

  struct KANN_EXPORT ActivationLayer final : public Layer
  {
  public:
    ActivationLayerDef def;

  public:
    KANN_EXPORT const LayerDef& get_def() const override;

    KANN_EXPORT tensor::Shape get_input_shape()  const override;
    KANN_EXPORT tensor::Shape get_output_shape() const override;

    KANN_EXPORT void initialize(std::default_random_engine& prng) override;

    KANN_EXPORT std::unordered_map<std::string, const Variable*> parameters_map() const override;
    KANN_EXPORT std::unordered_map<std::string, Variable*>       parameters_map()       override;

    KANN_EXPORT tensor::Tensor<float> forward(tensor::Tensor<float> inputs) override;
    KANN_EXPORT tensor::Tensor<float> backward(tensor::Tensor<float> output_gradients) override;
  };
}
