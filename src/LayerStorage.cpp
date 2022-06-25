#include <libkann/LayerStorage.hpp>

namespace kann
{
  std::vector<const Variable*> LayerStorage::get_parameters() const
  {
    std::vector<const Variable*> result;
    result |= ranges::actions::push_back(parameters | ranges::views::addressof);
    for(std::shared_ptr<const LayerStorage> sub_layer_storage : sub_layer_storages)
      result |= ranges::actions::push_back(sub_layer_storage->get_parameters());

    return result;
  }

  std::vector<const Variable*> LayerStorage::get_states() const
  {
    std::vector<const Variable*> result;
    result |= ranges::actions::push_back(states | ranges::views::addressof);
    for(std::shared_ptr<const LayerStorage> sub_layer_storage : sub_layer_storages)
      result |= ranges::actions::push_back(sub_layer_storage->get_states());

    return result;
  }

  std::vector<Variable*> LayerStorage::get_parameters()
  {
    std::vector<Variable*> result;
    result |= ranges::actions::push_back(parameters | ranges::views::addressof);
    for(std::shared_ptr<LayerStorage> sub_layer_storage : sub_layer_storages)
      result |= ranges::actions::push_back(sub_layer_storage->get_parameters());

    return result;
  }

  std::vector<Variable*> LayerStorage::get_states()
  {
    std::vector<Variable*> result;
    result |= ranges::actions::push_back(states | ranges::views::addressof);
    for(std::shared_ptr<LayerStorage> sub_layer_storage : sub_layer_storages)
      result |= ranges::actions::push_back(sub_layer_storage->get_states());

    return result;
  }
}
