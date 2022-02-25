#pragma once

#include <libkann/Layer.hpp>

namespace kann
{
  Ref<Layer> loadLayer(const std::string& filename);
  Ref<Layer> loadLayer(std::istream& is);
}
