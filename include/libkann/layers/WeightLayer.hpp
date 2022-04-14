#pragma once

#include <libkann/Layer.hpp>

namespace kann
{
  class WeightLayer : public Layer
  {
  public:
    WeightLayer() = default;
    WeightLayer(size_t input_size, size_t output_size);

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
      archive(m_input_size, m_output_size);
      archive(m_weight, m_bias);
    }

  private:
    size_t m_input_size, m_output_size;

  private:
    std::shared_ptr<const Tensor> m_weight;
    std::shared_ptr<const Tensor> m_bias;
  };

}

CEREAL_REGISTER_TYPE(kann::WeightLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::WeightLayer);
