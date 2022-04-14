#pragma once

#include <libkann/Layer.hpp>

namespace kann
{
  class ConvolutionalLayer final : public Layer
  {
  public:
    ConvolutionalLayer() = default;
    ConvolutionalLayer(size_t input_width, size_t input_height, size_t kernel_size, size_t input_channel_count, size_t output_channel_count);

  public:
    std::shared_ptr<Layer> clone() const override;
    void randomize(std::default_random_engine& engine) override;

  public:
    size_t input_size() const override;
    size_t output_size() const override;

  public:
    size_t parameters_count() const override;
    std::vector<size_t> parameter_sizes() const override;
    std::vector<std::shared_ptr<const Tensor>> get_parameters() const override;
    void set_parameters(std::vector<std::shared_ptr<const Tensor>> values) override;

  public:
    ProcessOutput process(ProcessInput input) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));

      archive(m_input_width, m_input_height);
      archive(m_kernel_size);
      archive(m_input_channel_count, m_output_channel_count);
      archive(m_kernels);
    }

  private:
    size_t m_input_width,  m_input_height;
    size_t m_kernel_size;
    size_t m_input_channel_count, m_output_channel_count;

  private:
    std::vector<std::shared_ptr<const Tensor>> m_kernels;
  };
}

CEREAL_REGISTER_TYPE(kann::ConvolutionalLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::ConvolutionalLayer);
