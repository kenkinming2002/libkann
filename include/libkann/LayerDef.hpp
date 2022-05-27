#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tag.hpp>

#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>

#include <memory>
#include <vector>
#include <random>

namespace kann
{
  struct Layer;
  struct LayerDef : public std::enable_shared_from_this<LayerDef>
  {
  public:
    // Question: How do we support tagging? Do we store tag in parent layer def or in child
    Tag tag = Tag::ALL;
    std::vector<layer_def_t> sub_layer_defs;

  public:
    virtual ~LayerDef() = default;

  public:
    virtual std::shared_ptr<Layer> create(std::default_random_engine& prng) const = 0;

  public:
    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;

  public:
    std::vector<Tag> parameters_tags() const;

  public:
    size_t parameters_all_count() const;
    size_t states_all_count() const;

  public:
    std::vector<size_t> parameters_all_sizes() const;
    std::vector<size_t> states_all_sizes() const;

  public:
    struct ProcessInput
    {
      variable_t variable;
      std::vector<variable_t> parameters;
      std::vector<variable_t> states;
    };

    struct ProcessOutput
    {
      variable_t variable;
      std::vector<variable_t> states;
    };

    virtual ProcessOutput process(ProcessInput input) const = 0;

  protected:
    virtual size_t parameters_count() const { return 0; }
    virtual size_t states_count()     const { return 0; }

  protected:
    virtual std::vector<size_t> parameters_sizes() const { return {}; }
    virtual std::vector<size_t> states_sizes()     const { return {}; }

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(sub_layer_defs);
    }
  };
}
