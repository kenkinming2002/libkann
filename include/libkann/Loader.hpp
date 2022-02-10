#pragma once

#include <libkann/Layer.hpp>

namespace kann
{
  std::shared_ptr<Layer> loadLayer(const std::string& filename);
  std::shared_ptr<Layer> loadLayer(std::istream& is);
}
