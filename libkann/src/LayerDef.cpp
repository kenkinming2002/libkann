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

  namespace
  {
    struct Info
    {
      std::string name;
      std::type_index type_index;

      LayerDef::save_t save;
      LayerDef::load_t load;
    };

    auto& name_map()
    {
      static std::unordered_map<std::string, Info> instance;
      return instance;
    }

    auto& type_index_map()
    {
      static std::unordered_map<std::type_index, Info> instance;
      return instance;
    }
  }

  void LayerDef::register_save_load(std::string name, const std::type_info& type_info, save_t save, load_t load)
  {
    auto type_index = std::type_index(type_info);
    auto info = Info{
      .name       = name,
      .type_index = type_index,
      .save       = save,
      .load       = load
    };
    name_map()      .emplace(name,       info);
    type_index_map().emplace(type_index, info);
  }

  YAML::Node LayerDef::save(std::shared_ptr<const LayerDef> layer)
  {
    auto type_index = std::type_index(typeid(*layer));
    const auto& info = type_index_map().at(type_index);

    auto node = info.save(layer);
    node["type"] = info.name;
    return node;
  }

  std::shared_ptr<const LayerDef> LayerDef::load(YAML::Node node)
  {
    auto name = node["type"].as<std::string>();
    const auto& info = name_map().at(name);

    node.remove("type");
    auto layer = info.load(node);
    return layer;
  }
}
