#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tag.hpp>

#include <range/v3/all.hpp>
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
    static void register_save_load(std::string name, const std::type_info& type_info, save_t save, load_t load);

    template<typename T>
    static void register_save_load(std::string name) { register_save_load(std::move(name), typeid(T), T::save, T::load); }

  public:
    static YAML::Node save(layer_def_t layer);
    static layer_def_t load(YAML::Node node);

  public:
    static layer_def_t load(const std::string& filename);
    static layer_def_t load(std::istream& is);

  public:
    // Question: How do we support tagging? Do we store tag in parent layer def or in child
    Tag tag = Tag::ALL;
    std::vector<layer_def_t> sub_layer_defs;

  public:
    virtual ~LayerDef() = default;

  public:
    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;

  public:
    struct Info
    {
      std::vector<size_t> parameter_sizes;
      std::vector<Tag>    parameter_tags;
      std::vector<size_t> parameter_indices;

      std::vector<size_t> state_sizes;
      std::vector<size_t> input_state_indices;
      std::vector<size_t> output_state_indices;

      void add_parameter(size_t size, Tag tag, size_t index)
      {
        parameter_sizes.push_back(size);
        parameter_tags.push_back(tag);
        parameter_indices.push_back(index);
      }

      void add_state(size_t size, size_t input_index, size_t output_index)
      {
        state_sizes.push_back(size);
        input_state_indices.push_back(input_index);
        output_state_indices.push_back(output_index);
      }

      void add_parameters(size_t size, Tag tag, const std::vector<size_t>& indices)
      {
        for(size_t index : indices)
          add_parameter(size, tag, index);
      }

      void add_states(size_t size, const std::vector<size_t>& input_indices, const std::vector<size_t>& output_indices)
      {
        for(const auto& [input_index, output_index] : ranges::views::zip(input_indices, output_indices))
          add_state(size, input_index, output_index);
      }
    };

  public:
    virtual std::shared_ptr<Layer> create(std::default_random_engine& prng) const = 0;
    virtual size_t process(Graph& graph, Info& info, size_t input_index) const = 0;
  };
}
