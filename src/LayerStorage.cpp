#include <libkann/LayerStorage.hpp>

#include <libkann/LayerDef.hpp>

#include <span>

#include <assert.h>

namespace kann
{
  typedef std::vector<Tensor> attr_t;
  typedef std::span<Tensor> view_t;
  typedef attr_t LayerStorage::* attr_ptr_t;

  static inline void set_attribute_all_impl(LayerStorage& layer, attr_ptr_t attr_ptr, view_t& view)
  {
    attr_t& attr = layer.*attr_ptr;

    std::copy_n(view.begin(), attr.size(), attr.begin());
    view = view.subspan(attr.size());
    for(auto& sub_layer : layer.sub_layer_storages)
      set_attribute_all_impl(*sub_layer, attr_ptr, view);
  }

  static inline void set_attribute_all(LayerStorage& layer, attr_ptr_t attr_ptr, attr_t attr)
  {
    view_t view(attr);
    set_attribute_all_impl(layer, attr_ptr, view);
    assert(view.empty());
  }

  void LayerStorage::set_parameters_all(std::vector<Tensor> values) { set_attribute_all(*this, &LayerStorage::parameters, values); }
  void LayerStorage::set_states_all(std::vector<Tensor> values)     { set_attribute_all(*this, &LayerStorage::states,     values); }

  static inline void get_attribute_all_impl(const LayerStorage& layer, attr_ptr_t attr_ptr, attr_t& target)
  {
    const attr_t& attr = layer.*attr_ptr;
    target |= ranges::actions::push_back(attr);
    for(auto& sub_layer : layer.sub_layer_storages)
      get_attribute_all_impl(*sub_layer, attr_ptr, target);
  }

  static inline attr_t get_attribute_all(const LayerStorage& layer, attr_ptr_t attr_ptr)
  {
    attr_t attr;
    get_attribute_all_impl(layer, attr_ptr, attr);
    return attr;
  }

  std::vector<Tensor> LayerStorage::get_parameters_all() const { return get_attribute_all(*this, &LayerStorage::parameters); }
  std::vector<Tensor> LayerStorage::get_states_all()     const { return get_attribute_all(*this, &LayerStorage::states); }
}
