#pragma once

#include <libkann/Export.hpp>

#include <libkann/LayerDef.hpp>

#include <libkann/Vec.hpp>

namespace kann
{
  class KANN_EXPORT ConvolutionalLayerDef : public LayerDef
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
    KANN_EXPORT Tensor forward(Layer& layer, Tensor inputs) const override;
    KANN_EXPORT Tensor backward(Layer& layer, Tensor output_gradients) const override;

  private:
    size_t m_input_channel_count, m_output_channel_count;
    Vec2 m_input_size, m_output_size, m_kernel_size;
  };
}
