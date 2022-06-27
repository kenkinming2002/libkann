#pragma once

#include <libkann/Export.hpp>
#include <libkann/Tag.hpp>
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
    KANN_EXPORT virtual tensor::Shape get_input_shape() const = 0;
    KANN_EXPORT virtual tensor::Shape get_output_shape() const = 0;

  public:
    KANN_EXPORT virtual tensor::Tensor<const float> forward(Layer& layer, tensor::Tensor<const float> inputs) const = 0;
    KANN_EXPORT virtual tensor::Tensor<const float> backward(Layer& layer, tensor::Tensor<const float> output_gradients) const = 0;

  protected:
    tensor::Tensor<const float> forward_helper(Layer& layer, tensor::Tensor<const float> inputs, const auto& impl) const
    {
      const size_t batch_size = inputs.shape().dimension(0);
      const tensor::Shape inputs_shape  = tensor::Shape::concat(tensor::Shape(batch_size), this->get_input_shape());
      const tensor::Shape outputs_shape = tensor::Shape::concat(tensor::Shape(batch_size), this->get_output_shape());

      assert(inputs.shape() == inputs_shape);
      tensor::Tensor<float> outputs = tensor::Tensor<float>::create(outputs_shape);
      return impl(layer, batch_size, std::move(inputs), std::move(outputs));
    }

    tensor::Tensor<const float> backward_helper(Layer& layer, tensor::Tensor<const float> output_gradients, const auto& impl) const
    {
      const size_t batch_size = output_gradients.shape().dimension(0);
      const tensor::Shape inputs_shape  = tensor::Shape::concat(tensor::Shape(batch_size), this->get_input_shape());
      const tensor::Shape outputs_shape = tensor::Shape::concat(tensor::Shape(batch_size), this->get_output_shape());

      assert(output_gradients.shape() == outputs_shape);
      tensor::Tensor<float> input_gradients = tensor::Tensor<float>::create(inputs_shape);
      return impl(layer, batch_size, std::move(output_gradients), std::move(input_gradients));
    }
  };
}
