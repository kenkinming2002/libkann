#include <libkann/LayerDef.hpp>

#include <range/v3/all.hpp>

#include <typeindex>
#include <fstream>

namespace kann
{
  void LayerDef::save(std::shared_ptr<const LayerDef> def, const std::string& filename)
  {
    std::ofstream of;
    of.exceptions(std::ofstream::badbit | std::ofstream::failbit);
    of.open(filename);
    save(std::move(def), of);
  }

  void LayerDef::save(std::shared_ptr<const LayerDef> def, std::ostream& os)
  {
    YAML::Node root = save(def);
    os << root;
  }

  std::shared_ptr<const LayerDef> LayerDef::load(const std::string& filename)
  {
    YAML::Node root = YAML::LoadFile(filename);
    return load(root);
  }

  std::shared_ptr<const LayerDef> LayerDef::load(std::istream& is)
  {
    YAML::Node root = YAML::Load(is);
    return load(root);
  }

  static auto& type_name_map()
  {
    static std::unordered_map<std::string, LayerDef::Info> instance;
    return instance;
  }

  static auto& type_index_map()
  {
    static std::unordered_map<std::type_index, LayerDef::Info> instance;
    return instance;
  }

  YAML::Node LayerDef::save(std::shared_ptr<const LayerDef> layer)
  {
    const auto& type_index = std::type_index(typeid(*layer));
    const auto& info = type_index_map().at(type_index);

    auto node = info.save(layer);
    node["type"] = info.type_name;
    return node;
  }

  std::shared_ptr<const LayerDef> LayerDef::load(YAML::Node node)
  {
    const auto& type_name = node["type"].as<std::string>();
    const auto& info = type_name_map().at(type_name);

    node.remove("type");
    auto layer = info.load(node);
    return layer;
  }

  void LayerDef::register_save_load(Info info)
  {
    type_index_map().emplace(info.type_index, info);
    type_name_map() .emplace(info.type_name,  info);
  }
}
