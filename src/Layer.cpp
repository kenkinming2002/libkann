#include <libkann/Layer.hpp>

#include <libkann/LayerDef.hpp>

#include <span>

#include <assert.h>

namespace kann
{
  typedef std::vector<tensor_t> attr_t;
  typedef std::span<tensor_t> view_t;
  typedef attr_t Layer::* attr_ptr_t;

  static inline size_t attribute_all_count(const Layer& layer, attr_ptr_t attr_ptr);
  static inline void set_attribute_all(Layer& layer, attr_ptr_t attr_ptr, attr_t attr);
  static inline attr_t get_attribute_all(const Layer& layer, attr_ptr_t attr_ptr, size_t count);
  static inline void set_attribute_all_impl(Layer& layer, attr_ptr_t attr_ptr, view_t& view);
  static inline void get_attribute_all_impl(const Layer& layer, attr_ptr_t attr_ptr, view_t& view);

  static inline size_t attribute_all_count(const Layer& layer, attr_ptr_t attr_ptr)
  {
    const attr_t& attr = layer.*attr_ptr;
    size_t size = attr.size();
    for(const auto& sub_layer : layer.sub_layers)
      size += attribute_all_count(*sub_layer, attr_ptr);
    return size;
  }

  static inline void set_attribute_all(Layer& layer, attr_ptr_t attr_ptr, attr_t attr)
  {
    view_t view(attr);
    set_attribute_all_impl(layer, attr_ptr, view);
    assert(view.empty());
  }

  static inline attr_t get_attribute_all(const Layer& layer, attr_ptr_t attr_ptr, size_t count)
  {
    attr_t attr(count);
    view_t view(attr);
    get_attribute_all_impl(layer, attr_ptr, view);
    return attr;
  }

  static inline void set_attribute_all_impl(Layer& layer, attr_ptr_t attr_ptr, view_t& view)
  {
    attr_t& attr = layer.*attr_ptr;

    std::copy_n(view.begin(), attr.size(), attr.begin());
    view = view.subspan(attr.size());
    for(auto& sub_layer : layer.sub_layers)
      set_attribute_all_impl(*sub_layer, attr_ptr, view);
  }

  static inline void get_attribute_all_impl(const Layer& layer, attr_ptr_t attr_ptr, view_t& view)
  {
    const attr_t& attr = layer.*attr_ptr;

    std::copy_n(attr.begin(), attr.size(), view.begin());
    view = view.subspan(attr.size());
    for(auto& sub_layer : layer.sub_layers)
      get_attribute_all_impl(*sub_layer, attr_ptr, view);
  }

  size_t Layer::parameters_all_count() const { return attribute_all_count(*this, &Layer::parameters); }
  size_t Layer::states_all_count()     const { return attribute_all_count(*this, &Layer::states); }

  void Layer::set_parameters_all(std::vector<tensor_t> values) { set_attribute_all(*this, &Layer::parameters, values); }
  void Layer::set_states_all(std::vector<tensor_t> values)     { set_attribute_all(*this, &Layer::states,     values); }

  std::vector<tensor_t> Layer::get_parameters_all() const { return get_attribute_all(*this, &Layer::parameters, parameters_all_count()); }
  std::vector<tensor_t> Layer::get_states_all()     const { return get_attribute_all(*this, &Layer::states,     states_all_count()); }
}
