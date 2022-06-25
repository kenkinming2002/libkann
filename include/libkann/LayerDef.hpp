#pragma once

#include <libkann/Export.hpp>
#include <libkann/Tag.hpp>
#include <libkann/Shape.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/LayerStorage.hpp>

#include <yaml-cpp/yaml.h>

#include <typeinfo>

#include <vector>
#include <random>

namespace kann
{
  struct Layer;
  struct LayerDef
  {
  public:
    using save_t = YAML::Node(*)(std::shared_ptr<const LayerDef>);
    using load_t = std::shared_ptr<const LayerDef>(*)(YAML::Node);

  public:
    KANN_EXPORT static void register_save_load(std::string name, const std::type_info& type_info, save_t save, load_t load);

    template<typename T>
    static void register_save_load(std::string name) { register_save_load(std::move(name), typeid(T), T::save, T::load); }

  public:
    KANN_EXPORT static YAML::Node save(std::shared_ptr<const LayerDef> layer);
    KANN_EXPORT static std::shared_ptr<const LayerDef> load(YAML::Node node);

  public:
    KANN_EXPORT static std::shared_ptr<const LayerDef> load(const std::string& filename);
    KANN_EXPORT static std::shared_ptr<const LayerDef> load(std::istream& is);

  public:
    // Question: How do we support tagging? Do we store tag in parent layer def or in child
    Tag tag = Tag::ALL;
    std::vector<std::shared_ptr<const LayerDef>> sub_layer_defs;

  public:
    KANN_EXPORT virtual ~LayerDef() = default;

  public:
    KANN_EXPORT virtual std::shared_ptr<LayerStorage> create(std::default_random_engine& prng) const = 0;

  public:
    KANN_EXPORT virtual Shape get_input_shape() const = 0;
    KANN_EXPORT virtual Shape get_output_shape() const = 0;

  public:
    KANN_EXPORT virtual Tensor<float> forward(Layer& layer, Tensor<float> inputs) const = 0;
    KANN_EXPORT virtual Tensor<float> backward(Layer& layer, Tensor<float> output_gradients) const = 0;
  };
}
