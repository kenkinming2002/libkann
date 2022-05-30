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

  std::vector<Tag> LayerDef::parameters_tags() const
  {
    auto parent_tags  = ranges::views::repeat_n(this->tag, this->parameters_count());
    auto childs_tags = sub_layer_defs
      | ranges::views::transform([](const auto& sub_layer_def) -> std::vector<Tag> { return sub_layer_def->parameters_tags(); })
      | ranges::views::cache1
      | ranges::views::join;

    return ranges::views::concat(parent_tags, childs_tags) | ranges::to_vector;
  }

  size_t LayerDef::parameters_all_count() const
  {
    size_t size = this->parameters_count();
    for(const auto& sub_layer_def : this->sub_layer_defs)
      size += sub_layer_def->parameters_all_count();
    return size;
  }

  size_t LayerDef::states_all_count() const
  {
    size_t size = this->states_count();
    for(const auto& sub_layer_def : this->sub_layer_defs)
      size += sub_layer_def->states_all_count();
    return size;
  }

  std::vector<size_t> LayerDef::parameters_all_sizes() const
  {
    std::vector<size_t> sizes = this->parameters_sizes();
    for(const auto& sub_layer_def : this->sub_layer_defs)
    {
      std::vector<size_t> sub_layer_sizes = sub_layer_def->parameters_all_sizes();
      sizes.insert(sizes.end(),
        std::move_iterator(sub_layer_sizes.begin()),
        std::move_iterator(sub_layer_sizes.end())
      );
    }
    return sizes;
  }

  std::vector<size_t> LayerDef::states_all_sizes() const
  {
    std::vector<size_t> sizes = this->states_sizes();
    for(const auto& sub_layer_def : this->sub_layer_defs)
    {
      std::vector<size_t> sub_layer_sizes = sub_layer_def->states_all_sizes();
      sizes.insert(sizes.end(),
        std::move_iterator(sub_layer_sizes.begin()),
        std::move_iterator(sub_layer_sizes.end())
      );
    }
    return sizes;
  }
}
