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

    weight.value.fill_normal(prng, 0.0, 1.0 / std::sqrt(input_shape.size()));
    bias.value.fill_normal(prng, 0.0, 1.0 / std::sqrt(input_shape.size()));

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
    return this->forward_helper(layer, std::move(inputs), [this](Layer& layer, size_t batch_size, Tensor<float> inputs, Tensor<float> outputs)
    {
      const size_t input_size  = this->get_input_shape().size();
      const size_t output_size = this->get_output_shape().size();

      const Variable& weight = layer.storage->parameters[0];
      const Variable& bias   = layer.storage->parameters[1];

      auto _inputs  = inputs .reshape(Shape{batch_size, input_size});
      auto _outputs = outputs.reshape(Shape{batch_size, output_size});

      auto _weight  = weight.value.reshape(Shape{input_size, output_size});
      auto _bias    = bias  .value.reshape(Shape{output_size});

      math::product(_outputs, _inputs, false, _weight, false);
      math::broadcast<1>(_outputs, { _bias }, math::Direction::LEFT, math::ADD);

      layer.saved_tensors.clear();
      layer.saved_tensors.reserve(1);
      layer.saved_tensors.push_back(std::move(inputs));

      return outputs;
    });
  }

  Tensor<float> DenseLayerDef::backward(Layer& layer, Tensor<float> output_gradients) const
  {
    return this->backward_helper(layer, std::move(output_gradients), [this](Layer& layer, size_t batch_size, Tensor<float> output_gradients, Tensor<float> input_gradients)
    {
      const size_t input_size  = this->get_input_shape().size();
      const size_t output_size = this->get_output_shape().size();

      Tensor<float> inputs = std::move(layer.saved_tensors[0]);
      Variable& weight = layer.storage->parameters[0];
      Variable& bias   = layer.storage->parameters[1];

      auto _inputs           = inputs          .reshape(Shape{batch_size, input_size});
      auto _input_gradients  = input_gradients .reshape(Shape{batch_size, input_size});
      auto _output_gradients = output_gradients.reshape(Shape{batch_size, output_size});

      auto _weight          = weight.value   .reshape(Shape{input_size, output_size});
      auto _weight_gradient = weight.gradient.reshape(Shape{input_size, output_size});

      auto _bias          = bias.value   .reshape(Shape{output_size});
      auto _bias_gradient = bias.gradient.reshape(Shape{output_size});

      math::product(_weight_gradient, _inputs,           true,  _output_gradients, false);
      math::product(_input_gradients, _output_gradients, false, _weight,           true);

      _bias_gradient.fill(0.0);
      math::reduce<1>(_bias_gradient, { _output_gradients }, math::Direction::LEFT, math::ADD);

      return input_gradients;
    });
  }
}
