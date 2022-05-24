#pragma once

#include <libkann/Tag.hpp>
#include <libkann/LayerDef.hpp>

namespace kann
{
  std::shared_ptr<const LayerDef> load_layer_def(const std::string& filename, Tag tag = Tag::ALL);
  std::shared_ptr<const LayerDef> load_layer_def(std::istream& is, Tag tag = Tag::ALL);
}
