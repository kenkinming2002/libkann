#include <libkann/layer_defs/Dense.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Math.hpp>
#include <libkann/Layer.hpp>

namespace kann
{
  YAML::Node DenseLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["input_shape"]  = Shape::to_vector(std::static_pointer_cast<const DenseLayerDef>(layer_def)->m_input_shape);
    node["output_shape"] = Shape::to_vector(std::static_pointer_cast<const DenseLayerDef>(layer_def)->m_output_shape);
    return node;
  }

  std::shared_ptr<const LayerDef> DenseLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<DenseLayerDef>();
    layer_def->m_input_shape  = Shape::from_vector(node["input_shape"].as<std::vector<size_t>>());
    layer_def->m_output_shape = Shape::from_vector(node["output_shape"].as<std::vector<size_t>>());
    return layer_def;
  }

  std::shared_ptr<LayerStorage> DenseLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer_storage = std::make_shared<LayerStorage>();

    Tensor<float> weight = Tensor<float>::create(Shape::concat(m_input_shape, m_output_shape));
    Tensor<float> bias   = Tensor<float>::create(m_output_shape);
    weight.as_ref().fill_normal(prng, 0.0, 1.0 / std::sqrt(m_input_shape.size()));
    bias.as_ref().fill_normal(prng, 0.0, 1.0 / std::sqrt(m_input_shape.size()));

    layer_storage->parameters.reserve(2);
    layer_storage->parameters.push_back(Variable{.value = std::move(weight)});
    layer_storage->parameters.push_back(Variable{.value = std::move(bias)});

    return layer_storage;
  }

  Shape DenseLayerDef::input_shape() const
  {
    return m_input_shape;
  }

  Shape DenseLayerDef::output_shape() const
  {
    return m_output_shape;
  }

  Tensor<float> DenseLayerDef::forward(Layer& layer, Tensor<float> inputs) const
  {
    const Variable& weight = layer.storage->parameters[0];
    const Variable& bias   = layer.storage->parameters[1];

    const size_t batch_size = inputs.as_ref().shape().dimension(0);
    Tensor<float> outputs = Tensor<float>::create(Shape::concat(Shape(batch_size), m_output_shape));
    math::product(outputs.as_ref(), inputs.as_const_ref(), false, weight.value.as_const_ref(), false);
    math::broadcast<1>(outputs.as_ref(), {bias.value.as_const_ref()}, math::Direction::LEFT, math::ADD);

    layer.saved_tensors.clear();
    layer.saved_tensors.reserve(1);
    layer.saved_tensors.push_back(std::move(inputs));

    return outputs;
  }

  Tensor<float> DenseLayerDef::backward(Layer& layer, Tensor<float> output_gradients) const
  {
    Variable& weight = layer.storage->parameters[0];
    Variable& bias   = layer.storage->parameters[1];
    const Tensor<float>& inputs = layer.saved_tensors[0];

    const size_t batch_size = output_gradients.as_ref().shape().dimension(0);
    Tensor<float> weight_gradient = Tensor<float>::create(Shape::concat(m_input_shape, m_output_shape));
    Tensor<float> bias_gradient   = Tensor<float>::create(m_output_shape);
    Tensor<float> inputs_gradient = Tensor<float>::create(Shape::concat(Shape(batch_size), m_input_shape));

    math::product(weight_gradient.as_ref(), inputs.as_const_ref(),           true,  output_gradients.as_const_ref(), false);
    math::product(inputs_gradient.as_ref(), output_gradients.as_const_ref(), false, weight.value.as_const_ref(),     true);

    bias_gradient.as_ref().fill(0.0);
    math::reduce<1>(bias_gradient.as_ref(), {output_gradients.as_const_ref()}, math::Direction::LEFT, math::ADD);

    weight.gradient = std::move(weight_gradient);
    bias.gradient   = std::move(bias_gradient);
    return inputs_gradient;
  }
}
