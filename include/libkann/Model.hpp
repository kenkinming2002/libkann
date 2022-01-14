#pragma once

#include <libkann/layers/Layer.hpp>

#include <cereal/types/vector.hpp>

namespace kann
{
  class Model : public Layer
  {
  public:
    Model() = default;
    Model(const Model& other);

  public:
    static std::unique_ptr<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    virtual void write_graphviz(std::ostream& os) const = 0;

  public:
    void randomize(std::default_random_engine& engine) override;
    void train(double learningRate, unsigned tags = TAG_ALL) override;

  public:
    std::vector<std::span<double>> params() override final;
    std::vector<std::span<const double>> params() const override final;

    std::vector<std::span<double>> paramsGradient() override final;
    std::vector<std::span<const double>> paramsGradient() const override final;

  protected:
    size_t addLayer(std::shared_ptr<Layer> layer);
    Layer& layer(size_t index);
    const Layer& layer(size_t index) const;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_layers);
    }

  private:
    std::vector<std::shared_ptr<Layer>> m_layers;
  };
}
