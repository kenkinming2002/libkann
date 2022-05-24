#include <libkann/Layer.hpp>

#include <libkann/LayerDef.hpp>

namespace kann
{
  typedef std::vector<std::shared_ptr<const Tensor>> attr_t;
  typedef std::span<std::shared_ptr<const Tensor>> view_t;
  typedef attr_t Layer::* attr_ptr_t;

  static inline void set_attribute_all(Layer& layer, attr_ptr_t attr_ptr, attr_t attr);
  static inline attr_t get_attribute_all(const Layer& layer, attr_ptr_t attr_ptr, size_t count);
  static inline void set_attribute_all_impl(Layer& layer, attr_ptr_t attr_ptr, view_t& view);
  static inline void get_attribute_all_impl(const Layer& layer, attr_ptr_t attr_ptr, view_t& view);

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

  void Layer::set_parameters_all(std::vector<std::shared_ptr<const Tensor>> values) { set_attribute_all(*this, &Layer::parameters, values); }
  void Layer::set_states_all(std::vector<std::shared_ptr<const Tensor>> values)     { set_attribute_all(*this, &Layer::states,     values); }

  std::vector<std::shared_ptr<const Tensor>> Layer::get_parameters_all() const { return get_attribute_all(*this, &Layer::parameters, def->parameters_all_count()); }
  std::vector<std::shared_ptr<const Tensor>> Layer::get_states_all()     const { return get_attribute_all(*this, &Layer::states,     def->states_all_count()); }
}
