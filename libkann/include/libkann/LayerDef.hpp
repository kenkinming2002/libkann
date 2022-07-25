#pragma once

#include <libkann/Export.hpp>

#include <memory>

namespace kann
{
  struct Layer;
  struct LayerDef
  {
  public:
    KANN_EXPORT virtual ~LayerDef() = default;
    KANN_EXPORT virtual std::shared_ptr<Layer> create() const = 0;
  };
}
