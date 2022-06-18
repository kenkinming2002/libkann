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

  void LayerDef::Info::add_parameter(Shape shape, Tag tag, size_t index)
  {
    parameter_shapes.push_back(shape);
    parameter_tags.push_back(tag);
    parameter_indices.push_back(index);
  }

  void LayerDef::Info::add_state(Shape shape, size_t input_index, size_t output_index)
  {
    state_shapes.push_back(shape);
    input_state_indices.push_back(input_index);
    output_state_indices.push_back(output_index);
  }

  void LayerDef::Info::add_parameters(Shape shape, Tag tag, const std::vector<size_t>& indices)
  {
    for(size_t index : indices)
      add_parameter(shape, tag, index);
  }

  void LayerDef::Info::add_states(Shape shape, const std::vector<size_t>& input_indices, const std::vector<size_t>& output_indices)
  {
    for(const auto& [input_index, output_index] : ranges::views::zip(input_indices, output_indices))
      add_state(shape, input_index, output_index);
  }
}
