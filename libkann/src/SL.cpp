#include <libkann/SL.hpp>

#include <libkann/LayerDef.hpp>
#include <libkann/Layer.hpp>

#include <libtensor/SL.hpp>

#include <filesystem>
#include <fstream>
#include <unordered_map>

#include <fmt/core.h>

namespace kann
{
  void save_layer_def(std::shared_ptr<const LayerDef> def, const std::string& filename)
  {
    std::ofstream of;
    of.exceptions(std::ofstream::badbit | std::ofstream::failbit);
    of.open(filename);
    YAML::Node root = save_layer_def(def);
    of << root;
  }

  std::shared_ptr<const LayerDef> load_layer_def(const std::string& filename)
  {
    YAML::Node root = YAML::LoadFile(filename);
    return load_layer_def(root);
  }

  static auto& type_name_map()
  {
    static std::unordered_map<std::string, LayerDefInfo> instance;
    return instance;
  }

  static auto& type_index_map()
  {
    static std::unordered_map<std::type_index, LayerDefInfo> instance;
    return instance;
  }

  YAML::Node save_layer_def(std::shared_ptr<const LayerDef> layer)
  {
    const auto& type_index = std::type_index(typeid(*layer));
    const auto& info = type_index_map().at(type_index);

    auto node = info.save(layer);
    node["type"] = info.type_name;
    return node;
  }

  std::shared_ptr<const LayerDef> load_layer_def(YAML::Node node)
  {
    const auto& type_name = node["type"].as<std::string>();
    const auto& info = type_name_map().at(type_name);

    node.remove("type");
    auto layer = info.load(node);
    return layer;
  }

  void layer_def_sl_register(LayerDefInfo info)
  {
    type_index_map().emplace(info.type_index, info);
    type_name_map() .emplace(info.type_name,  info);
  }

  void layer_save_parameters(const Layer& layer, const std::string& dirname, bool include_gradient)
  {
    std::filesystem::create_directories(dirname);
    for(const auto& [name, parameter] : layer.parameters_map())
    {
      tensor::save_tensor(parameter->value,    fmt::format("{}/{}.value",    dirname, name));
      if(!include_gradient) continue;
      tensor::save_tensor(parameter->gradient, fmt::format("{}/{}.gradient", dirname, name));
    }
  }

  void layer_load_parameters(Layer& layer, const std::string& dirname, bool include_gradient)
  {
    for(const auto& [name, parameter] : layer.parameters_map())
    {
      parameter->value    = tensor::load_tensor(fmt::format("{}/{}.value",    dirname, name));
      if(!include_gradient) continue;
      parameter->gradient = tensor::load_tensor(fmt::format("{}/{}.gradient", dirname, name));
    }
  }
}
