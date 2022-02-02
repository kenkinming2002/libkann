#pragma once

#include <libkann/NewLayer.hpp>

namespace kann
{
  class RecurrentLayer : public NewLayer
  {
  public:
    RecurrentLayer(size_t memory);

  public:
    void addLayer(std::shared_ptr<const NewLayer> layer);

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    std::vector<NewParameter> parameters() const override;
    std::vector<NewParameter> stateParameters() const override;

    LayerVariable operator()(Scope scope, LayerVariable) const override;

  private:
    static Scope layerScope(size_t i)
    {
      return Scope("layer"+std::to_string(i));
    }

  private:
    size_t m_memory;
    std::vector<std::shared_ptr<const NewLayer>> m_layers;
  };
}

