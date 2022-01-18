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
    virtual void write_graphviz(std::ostream& os) const = 0;

  // Layer parameters
  public:
    std::vector<std::shared_ptr<const Variable>> parametersVariables() const override;
    std::vector<std::shared_ptr<const Tensor>> parameters() const override;
    void parameters(std::vector<std::shared_ptr<const Tensor>> parameters) override;

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

  std::unique_ptr<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);
}
