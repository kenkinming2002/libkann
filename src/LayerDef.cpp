#include <libkann/LayerDef.hpp>

#include <range/v3/all.hpp>

#include <typeindex>
#include <iostream>

namespace kann
{
  static auto& save_map()
  {
    static std::unordered_map<std::type_index, LayerDef::save_t> instance;
    return instance;
  }

  static auto& load_map()
  {
    static std::unordered_map<std::string, LayerDef::load_t> instance;
    return instance;
  }

  void LayerDef::register_save_load(std::string name, const std::type_info& type_info, save_t save, load_t load)
  {
    save_map().emplace(std::type_index(type_info), save);
    load_map().emplace(std::move(name), load);
  }

  YAML::Node LayerDef::save(layer_def_t layer)
  {
    auto type_index = std::type_index(typeid(*layer));
    return save_map().at(type_index)(layer);
  }

  layer_def_t LayerDef::load(YAML::Node node)
  {
    auto name = node["type"].as<std::string>();
    return load_map().at(name)(node);
  }

  layer_def_t LayerDef::load(const std::string& filename)
  {
    YAML::Node root = YAML::LoadFile(filename);
    return load(root);
  }

  layer_def_t LayerDef::load(std::istream& is)
  {
    YAML::Node root = YAML::Load(is);
    return load(root);
  }
}
