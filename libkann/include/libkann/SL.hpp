#pragma once

#include <libkann/Export.hpp>

#include <yaml-cpp/yaml.h>

#include <memory>

#include <typeinfo>
#include <typeindex>

namespace kann
{
  struct LayerDef;
  struct Layer;

  // LayerDef save/load

  KANN_EXPORT void save_layer_def(const LayerDef& def, const std::string& filename);
  KANN_EXPORT std::unique_ptr<LayerDef> load_layer_def(const std::string& filename);

  KANN_EXPORT YAML::Node save_layer_def(const LayerDef& layer);
  KANN_EXPORT std::unique_ptr<LayerDef> load_layer_def(YAML::Node node);

  // LayerDef save/load specializaton
  // Again, ideally, we would have reflection support in c++, but we do not

  template<typename T> YAML::Node save_layer_def_impl(const T& def);
  template<typename T> T load_layer_def_impl(const YAML::Node& node);

  struct LayerDefInfo
  {
    std::string     type_name;
    std::type_index type_index;

    YAML::Node(*save)(const LayerDef& def);
    std::unique_ptr<LayerDef>(*load)(const YAML::Node& node);
  };

  KANN_EXPORT void layer_def_sl_register(LayerDefInfo info);
  template<typename T> void layer_def_sl_register(std::string name)
  {
    layer_def_sl_register(LayerDefInfo{
      .type_name  = std::move(name),
      .type_index = std::type_index(typeid(T)),
      .save = [](const LayerDef& def)     -> YAML::Node                { return save_layer_def_impl<T>(static_cast<const T&>(def));    },
      .load = [](const YAML::Node& node)  -> std::unique_ptr<LayerDef> { return std::make_unique<T>(load_layer_def_impl<T>(node)); },
    });
  }

  // Layer save/load
  KANN_EXPORT void save_layer_parameters(const Layer& layer, const std::string& dirname, bool include_gradient);
  KANN_EXPORT void load_layer_parameters(Layer& layer, const std::string& dirname, bool include_gradient);

  KANN_EXPORT void save_layer(const Layer& layer, const std::string& dirname, bool include_gradient);
  KANN_EXPORT std::unique_ptr<Layer> load_layer(const std::string& dirname, bool include_gradient);
}
