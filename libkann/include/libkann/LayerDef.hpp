#pragma once

#include <libkann/Export.hpp>

#include <memory>

namespace kann
{
  struct Layer;
  struct LayerDef
  {
  public:
    LIBKANN_EXPORT virtual ~LayerDef() = default;
    LIBKANN_EXPORT virtual std::unique_ptr<Layer> create() const = 0;
  };
}
