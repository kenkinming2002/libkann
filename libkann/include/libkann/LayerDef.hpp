#pragma once

#include <libkann/Export.hpp>
#include <libtensor/Shape.hpp>
#include <libtensor/Tensor.hpp>

#include <yaml-cpp/yaml.h>

#include <typeinfo>
#include <typeindex>

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
    KANN_EXPORT static YAML::Node save(std::shared_ptr<const LayerDef> layer);
    KANN_EXPORT static std::shared_ptr<const LayerDef> load(YAML::Node node);

  public:
    template<typename T> static YAML::Node save_impl(const T& def);
    template<typename T> static T load_impl(const YAML::Node& node);

  public:
    struct Info
    {
      std::string     type_name;
      std::type_index type_index;

      YAML::Node(*save)(const std::shared_ptr<const LayerDef>& def);
      std::shared_ptr<const LayerDef>(*load)(const YAML::Node& node);
    };
    KANN_EXPORT static void register_save_load(Info info);

    template<typename T>
    static void register_save_load(std::string name)
    {
      register_save_load(Info{
        .type_name  = std::move(name),
        .type_index = std::type_index(typeid(T)),
        .save = [](const std::shared_ptr<const LayerDef>& def) -> YAML::Node                      { return save_impl<T>(static_cast<const T&>(*def));    },
        .load = [](const YAML::Node& node)                     -> std::shared_ptr<const LayerDef> { return std::make_shared<const T>(load_impl<T>(node)); },
      });
    }

  public:
    KANN_EXPORT virtual ~LayerDef() = default;
    KANN_EXPORT virtual std::shared_ptr<Layer> create() const = 0;
  };
}
