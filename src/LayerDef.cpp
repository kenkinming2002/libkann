#include <libkann/LayerDef.hpp>

#include <range/v3/all.hpp>

namespace kann
{
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
