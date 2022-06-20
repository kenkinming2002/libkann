#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>
#include <libkann/Tag.hpp>
#include <libkann/Shape.hpp>

#include <yaml-cpp/yaml.h>

#include <typeinfo>

#include <vector>
#include <random>

namespace kann
{
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
    KANN_EXPORT virtual Shape input_shape() const = 0;
    KANN_EXPORT virtual Shape output_shape() const = 0;

  public:
    struct Info
    {
      std::vector<Shape>  parameter_shapes;
      std::vector<Tag>    parameter_tags;
      std::vector<size_t> parameter_indices;

      std::vector<Shape> state_shapes;
      std::vector<size_t> input_state_indices;
      std::vector<size_t> output_state_indices;

      KANN_EXPORT void add_parameter(Shape shape, Tag tag, size_t index);
      KANN_EXPORT void add_state(Shape shape, size_t input_index, size_t output_index);

      KANN_EXPORT void add_parameters(Shape shape, Tag tag, const std::vector<size_t>& indices);
      KANN_EXPORT void add_states(Shape shape, const std::vector<size_t>& input_indices, const std::vector<size_t>& output_indices);
    };

  public:
    KANN_EXPORT virtual std::shared_ptr<LayerStorage> create(std::default_random_engine& prng) const = 0;
    KANN_EXPORT virtual size_t batch_process(Graph& graph, Info& info, size_t batch_size, size_t input_index) const { assert(false && "Unimplemented"); }
  };
}
