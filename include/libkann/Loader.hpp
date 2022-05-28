#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tag.hpp>

#include <iosfwd>
#include <string>

namespace kann
{
  layer_def_t load_layer_def(const std::string& filename);
  layer_def_t load_layer_def(std::istream& is);
}
