#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Shape.hpp>
#include <libtensor/Tensor.hpp>

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
    KANN_EXPORT virtual ~LayerDef() = default;
    KANN_EXPORT virtual std::shared_ptr<Layer> create() const { assert(false && "Unimplemented"); };
  };
}
