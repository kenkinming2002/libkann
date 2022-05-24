#pragma once

#include <libkann/LayerDef.hpp>

namespace kann
{
  class ConvolutionalLayerDef : public LayerDef
  {
  public:
    ConvolutionalLayerDef() = default;
    ConvolutionalLayerDef(size_t input_width, size_t input_height, size_t kernel_size, size_t input_channel_count, size_t output_channel_count);

  public:
    std::shared_ptr<Layer> create(std::default_random_engine& prng) const override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  protected:
    size_t parameters_count() const override;
    std::vector<size_t> parameters_sizes() const override;

  private:
    size_t m_input_width,  m_input_height;
    size_t m_kernel_size;
    size_t m_input_channel_count, m_output_channel_count;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<LayerDef>(this));

      archive(m_input_width, m_input_height);
      archive(m_kernel_size);
      archive(m_input_channel_count, m_output_channel_count);
    }
  };
}

CEREAL_REGISTER_TYPE(kann::ConvolutionalLayerDef);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::LayerDef, kann::ConvolutionalLayerDef);
