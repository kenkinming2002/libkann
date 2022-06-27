#include <libkann/layer_defs/SoftMax.hpp>

#include <libtensor/Tensor.hpp>
#include <libtensor/Math.hpp>
#include <libkann/Layer.hpp>

#include <fmt/core.h>

namespace kann
{
  YAML::Node SoftMaxLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["shape"] = tensor::Shape::to_vector(std::static_pointer_cast<const SoftMaxLayerDef>(layer_def)->shape);
    return node;
  }

  std::shared_ptr<const LayerDef> SoftMaxLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<SoftMaxLayerDef>();
    layer_def->shape = tensor::Shape::from_vector(node["shape"].as<std::vector<size_t>>());
    return layer_def;
  }

  std::shared_ptr<LayerStorage> SoftMaxLayerDef::create(std::default_random_engine& prng) const
  {
    return std::make_shared<LayerStorage>();
  }

  tensor::Shape SoftMaxLayerDef::get_input_shape() const
  {
    return shape;
  }

  tensor::Shape SoftMaxLayerDef::get_output_shape() const
  {
    return shape;
  }

  tensor::Tensor<const float> SoftMaxLayerDef::forward(Layer& layer, tensor::Tensor<const float> inputs) const
  {
    return this->forward_helper(layer, std::move(inputs), [this](Layer& layer, size_t batch_size, tensor::Tensor<const float> inputs, tensor::Tensor<float> outputs)
    {
      const size_t size = this->shape.size();

      auto _inputs  = inputs .reshape(tensor::Shape{batch_size, size});
      auto _outputs = outputs.reshape(tensor::Shape{batch_size, size});

      tensor::math::transform<1>(_outputs.flatten(), { _inputs.flatten() }, [](double /*output*/, double input) { return std::exp(input); });

      // Batch normalization
      tensor::Tensor<float> factors = tensor::Tensor<float>::create(tensor::Shape(batch_size));
      factors.fill(0.0);
      tensor::math::reduce<1>   (factors,  { _outputs }, tensor::math::Direction::RIGHT, tensor::math::ADD);
      tensor::math::broadcast<1>(_outputs, { factors },  tensor::math::Direction::RIGHT, tensor::math::DIV);

      // It is actually better to save outouts
      layer.saved_tensors.clear();
      layer.saved_tensors.reserve(1);
      layer.saved_tensors.push_back(outputs);
      return outputs;
    });
  }

  tensor::Tensor<const float> SoftMaxLayerDef::backward(Layer& layer, tensor::Tensor<const float> output_gradients) const
  {
    return this->backward_helper(layer, std::move(output_gradients), [this](Layer& layer, size_t batch_size, tensor::Tensor<const float> output_gradients, tensor::Tensor<float> input_gradients)
    {
      const size_t size = this->shape.size();

      tensor::Tensor<const float> outputs = std::move(layer.saved_tensors[0]);

      auto _outputs          = outputs         .reshape(tensor::Shape{batch_size, size});
      auto _input_gradients  = input_gradients .reshape(tensor::Shape{batch_size, size});
      auto _output_gradients = output_gradients.reshape(tensor::Shape{batch_size, size});

      _input_gradients.fill(0.0f);
      for(size_t k=0; k<batch_size; ++k)
        for(size_t i=0; i<size; ++i)
          for(size_t j=0; j<size; ++j)
          {
            if(i == j)
              _input_gradients(k, i) += _output_gradients(k, j) * _outputs(k, j) * (1 - _outputs(k, i));
            else
              _input_gradients(k, i) -= _output_gradients(k, j) * _outputs(k, j) * _outputs(k, i);
          }

      return input_gradients;
    });
  }
}

