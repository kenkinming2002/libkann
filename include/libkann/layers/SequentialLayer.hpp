#pragma once

#include <libkann/NewLayer.hpp>

namespace kann
{
  class SequentialLayer : public NewLayer
  {
  public:
    void addLayer(std::shared_ptr<const NewLayer> layer);

  public:
    size_t inputSize() const override;
    size_t outputSize() const override;

  public:
    std::vector<NewParameter> parameters() const override;
    std::vector<NewParameter> stateParameters() const override;

    LayerVariable operator()(LayerVariable) const override;

  private:
    std::vector<std::shared_ptr<const NewLayer>> m_layers;
  };
}
