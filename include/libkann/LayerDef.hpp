#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Tag.hpp>
#include <libkann/Shape.hpp>

#include <yaml-cpp/yaml.h>

#include <typeinfo>

#include <vector>
#include <random>

namespace kann
{
  struct Layer;
  struct LayerDef : public std::enable_shared_from_this<LayerDef>
  {
  public:
    using save_t = YAML::Node(*)(layer_def_t);
    using load_t = layer_def_t(*)(YAML::Node);

  public:
    KANN_EXPORT static void register_save_load(std::string name, const std::type_info& type_info, save_t save, load_t load);

    template<typename T>
    static void register_save_load(std::string name) { register_save_load(std::move(name), typeid(T), T::save, T::load); }

  public:
    KANN_EXPORT static YAML::Node save(layer_def_t layer);
    KANN_EXPORT static layer_def_t load(YAML::Node node);

  public:
    KANN_EXPORT static layer_def_t load(const std::string& filename);
    KANN_EXPORT static layer_def_t load(std::istream& is);

  public:
    // Question: How do we support tagging? Do we store tag in parent layer def or in child
    Tag tag = Tag::ALL;
    std::vector<layer_def_t> sub_layer_defs;

  public:
    KANN_EXPORT virtual ~LayerDef() = default;

  public:
    KANN_EXPORT virtual std::shared_ptr<LayerStorage> create(std::default_random_engine& prng) const = 0;

  public:
    KANN_EXPORT virtual Shape input_shape() const = 0;
    KANN_EXPORT virtual Shape output_shape() const = 0;

  public:
    KANN_EXPORT virtual Tensor forward(Layer& layer, Tensor input) const = 0;
    KANN_EXPORT virtual Tensor backward(Layer& layer, Tensor output_gradient) const = 0;
  };
}
