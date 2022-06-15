#include <libkann/Layer.hpp>

#include <libkann/LayerDef.hpp>

#include <span>

#include <assert.h>

namespace kann
{
  typedef std::vector<Tensor> attr_t;
  typedef std::span<Tensor> view_t;
  typedef attr_t Layer::* attr_ptr_t;

  static inline void set_attribute_all_impl(Layer& layer, attr_ptr_t attr_ptr, view_t& view)
  {
    attr_t& attr = layer.*attr_ptr;

    std::copy_n(view.begin(), attr.size(), attr.begin());
    view = view.subspan(attr.size());
    for(auto& sub_layer : layer.sub_layers)
      set_attribute_all_impl(*sub_layer, attr_ptr, view);
  }

  static inline void set_attribute_all(Layer& layer, attr_ptr_t attr_ptr, attr_t attr)
  {
    view_t view(attr);
    set_attribute_all_impl(layer, attr_ptr, view);
    assert(view.empty());
  }

  void Layer::set_parameters_all(std::vector<Tensor> values) { set_attribute_all(*this, &Layer::parameters, values); }
  void Layer::set_states_all(std::vector<Tensor> values)     { set_attribute_all(*this, &Layer::states,     values); }

  static inline void get_attribute_all_impl(const Layer& layer, attr_ptr_t attr_ptr, attr_t& target)
  {
    const attr_t& attr = layer.*attr_ptr;
    target |= ranges::actions::push_back(attr);
    for(auto& sub_layer : layer.sub_layers)
      get_attribute_all_impl(*sub_layer, attr_ptr, target);
  }

  static inline attr_t get_attribute_all(const Layer& layer, attr_ptr_t attr_ptr)
  {
    attr_t attr;
    get_attribute_all_impl(layer, attr_ptr, attr);
    return attr;
  }

  std::vector<Tensor> Layer::get_parameters_all() const { return get_attribute_all(*this, &Layer::parameters); }
  std::vector<Tensor> Layer::get_states_all()     const { return get_attribute_all(*this, &Layer::states); }
}
