#include <libkann/layer_defs/Dense.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Math.hpp>
#include <libkann/Layer.hpp>

namespace kann
{
  YAML::Node DenseLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["input_shape"]  = Shape::to_vector(std::static_pointer_cast<const DenseLayerDef>(layer_def)->input_shape);
    node["output_shape"] = Shape::to_vector(std::static_pointer_cast<const DenseLayerDef>(layer_def)->output_shape);
    return node;
  }

  std::shared_ptr<const LayerDef> DenseLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<DenseLayerDef>();
    layer_def->input_shape  = Shape::from_vector(node["input_shape"].as<std::vector<size_t>>());
    layer_def->output_shape = Shape::from_vector(node["output_shape"].as<std::vector<size_t>>());
    return layer_def;
  }

  std::shared_ptr<LayerStorage> DenseLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer_storage = std::make_shared<LayerStorage>();

    Variable weight = Variable::create(Shape::concat(input_shape, output_shape));
    Variable bias   = Variable::create(output_shape);

    weight.value.as_ref().fill_normal(prng, 0.0, 1.0 / std::sqrt(input_shape.size()));
    bias.value.as_ref().fill_normal(prng, 0.0, 1.0 / std::sqrt(input_shape.size()));

    layer_storage->parameters.reserve(2);
    layer_storage->parameters.push_back(std::move(weight));
    layer_storage->parameters.push_back(std::move(bias));

    return layer_storage;
  }

  Shape DenseLayerDef::get_input_shape() const
  {
    return input_shape;
  }

  Shape DenseLayerDef::get_output_shape() const
  {
    return output_shape;
  }

  Tensor<float> DenseLayerDef::forward(Layer& layer, Tensor<float> inputs) const
  {
    return this->forward_helper(layer, std::move(inputs), [](Layer& layer, size_t batch_size, Tensor<float> inputs, Tensor<float> outputs)
    {
      const Variable& weight = layer.storage->parameters[0];
      const Variable& bias   = layer.storage->parameters[1];

      math::product(outputs.as_ref(), inputs.as_const_ref(), false, weight.value.as_const_ref(), false);
      math::broadcast<1>(outputs.as_ref(), {bias.value.as_const_ref()}, math::Direction::LEFT, math::ADD);

      layer.saved_tensors.clear();
      layer.saved_tensors.reserve(1);
      layer.saved_tensors.push_back(std::move(inputs));

      return outputs;
    });
  }

  Tensor<float> DenseLayerDef::backward(Layer& layer, Tensor<float> output_gradients) const
  {
    return this->backward_helper(layer, std::move(output_gradients), [](Layer& layer, size_t batch_size, Tensor<float> output_gradients, Tensor<float> input_gradients)
    {
      const Tensor<float>& inputs = layer.saved_tensors[0];
      Variable& weight = layer.storage->parameters[0];
      Variable& bias   = layer.storage->parameters[1];

      math::product(weight.gradient.as_ref(), inputs.as_const_ref(),           true,  output_gradients.as_const_ref(), false);
      math::product(input_gradients.as_ref(), output_gradients.as_const_ref(), false, weight.value.as_const_ref(),     true);

      bias.gradient.as_ref().fill(0.0);
      math::reduce<1>(bias.gradient.as_ref(), {output_gradients.as_const_ref()}, math::Direction::LEFT, math::ADD);

      return input_gradients;
    });
  }
}
