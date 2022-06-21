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
    layer_storage->parameters = {
      Variable{.value = MutableTensor::normal(Shape::concat(m_input_shape, m_output_shape), prng, 0.0, 1.0 / std::sqrt(m_input_shape.size()))},
      Variable{.value = MutableTensor::normal(m_output_shape,                               prng, 0.0, 1.0 / std::sqrt(m_input_shape.size()))}
    };
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

  Tensor DenseLayerDef::forward(Layer& layer, Tensor inputs) const
  {
    const Variable& weight = layer.storage->parameters[0];
    const Variable& bias   = layer.storage->parameters[1];
    layer.saved_tensors = { inputs };

    const size_t batch_size = inputs.shape().dimension(0);
    MutableTensor outputs = MutableTensor::create(Shape::concat(Shape(batch_size), m_output_shape));
    math::product(inputs.as_ref(), false, weight.value.as_const().as_ref(), false, outputs.as_ref());
    math::broadcast(bias.value.as_const().as_ref(), outputs.as_ref(), math::Direction::LEFT, math::ADD);
    return outputs.as_const();
  }

  Tensor DenseLayerDef::backward(Layer& layer, Tensor output_gradients) const
  {
    Variable& weight = layer.storage->parameters[0];
    Variable& bias   = layer.storage->parameters[1];
    const Tensor& inputs = layer.saved_tensors[0];

    const size_t batch_size = output_gradients.shape().dimension(0);
    MutableTensor weight_gradient = MutableTensor::create(Shape::concat(m_input_shape, m_output_shape));
    MutableTensor bias_gradient   = MutableTensor::create(m_output_shape);
    MutableTensor inputs_gradient = MutableTensor::create(Shape::concat(Shape(batch_size), m_input_shape));

    math::product(inputs.as_ref(),           true,  output_gradients.as_ref(),        false, weight_gradient.as_ref());
    math::product(output_gradients.as_ref(), false, weight.value.as_const().as_ref(), true,  inputs_gradient.as_ref());

    bias_gradient.fill(0.0);
    math::reduce(output_gradients.as_ref(), bias_gradient.as_ref(), math::Direction::LEFT, math::ADD);

    weight.gradient = std::move(weight_gradient).as_const();
    bias.gradient   = std::move(bias_gradient).as_const();
    return std::move(inputs_gradient).as_const();
  }
}
