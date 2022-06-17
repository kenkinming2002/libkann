#pragma once

#include <libkann/LayerDef.hpp>

#include <libkann/Vec.hpp>

namespace kann
{
  class ConvolutionalLayerDef : public LayerDef
  {
  public:
    static YAML::Node save(layer_def_t layer_def);
    static layer_def_t load(YAML::Node node);

  public:
    ConvolutionalLayerDef() = default;
    ConvolutionalLayerDef(size_t input_channel_count, size_t output_channel_count, Vec2 input_size, Vec2 output_size, Vec2 kernel_size);

  public:
    Shape input_shape() const override;
    Shape output_shape() const override;

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;
    size_t process(Graph& graph, Info& info, size_t input_index) const override;

  private:
    size_t m_input_channel_count, m_output_channel_count;
    Vec2 m_input_size, m_output_size, m_kernel_size;
  };
}
