#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Shape.hpp>
#include <libtensor/Tensor.hpp>
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
    KANN_EXPORT static void save(std::shared_ptr<const LayerDef> def, const std::string& filename);
    KANN_EXPORT static void save(std::shared_ptr<const LayerDef> def, std::ostream& os);

    KANN_EXPORT static std::shared_ptr<const LayerDef> load(const std::string& filename);
    KANN_EXPORT static std::shared_ptr<const LayerDef> load(std::istream& is);

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
    std::vector<std::shared_ptr<const LayerDef>> sub_layer_defs;

  public:
    KANN_EXPORT virtual ~LayerDef() = default;

  public:
    KANN_EXPORT virtual std::shared_ptr<LayerStorage> create(std::default_random_engine& prng) const = 0;

  public:
    KANN_EXPORT virtual tensor::Shape get_input_shape() const = 0;
    KANN_EXPORT virtual tensor::Shape get_output_shape() const = 0;

  public:
    KANN_EXPORT virtual tensor::Tensor<float> forward(Layer& layer, tensor::Tensor<float> inputs) const = 0;
    KANN_EXPORT virtual tensor::Tensor<float> backward(Layer& layer, tensor::Tensor<float> output_gradients) const = 0;
  };
}
