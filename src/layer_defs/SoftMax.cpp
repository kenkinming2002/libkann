#include <libkann/layer_defs/SoftMax.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Math.hpp>
#include <libkann/Layer.hpp>

#include <fmt/core.h>

namespace kann
{
  YAML::Node SoftMaxLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["shape"] = Shape::to_vector(std::static_pointer_cast<const SoftMaxLayerDef>(layer_def)->shape);
    return node;
  }

  std::shared_ptr<const LayerDef> SoftMaxLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<SoftMaxLayerDef>();
    layer_def->shape = Shape::from_vector(node["shape"].as<std::vector<size_t>>());
    return layer_def;
  }

  std::shared_ptr<LayerStorage> SoftMaxLayerDef::create(std::default_random_engine& prng) const
  {
    return std::make_shared<LayerStorage>();
  }

  Shape SoftMaxLayerDef::get_input_shape() const
  {
    return shape;
  }

  Shape SoftMaxLayerDef::get_output_shape() const
  {
    return shape;
  }

  Tensor<float> SoftMaxLayerDef::forward(Layer& layer, Tensor<float> inputs) const
  {
    return this->forward_helper(layer, std::move(inputs), [this](Layer& layer, size_t batch_size, Tensor<float> inputs, Tensor<float> outputs)
    {
      const size_t size = this->shape.size();

      auto _inputs  = inputs .reshape(Shape{batch_size, size});
      auto _outputs = outputs.reshape(Shape{batch_size, size});

      math::transform<1>(_outputs.flatten(), { _inputs.flatten() }, [](double /*output*/, double input) { return std::exp(input); });

      // Batch normalization
      Tensor<float> factors = Tensor<float>::create(Shape(batch_size));
      factors.fill(0.0);
      math::reduce<1>   (factors,  { _outputs }, math::Direction::RIGHT, math::ADD);
      math::broadcast<1>(_outputs, { factors },  math::Direction::RIGHT, math::DIV);

      // It is actually better to save outouts
      layer.saved_tensors.clear();
      layer.saved_tensors.reserve(1);
      layer.saved_tensors.push_back(outputs);
      return outputs;
    });
  }

  Tensor<float> SoftMaxLayerDef::backward(Layer& layer, Tensor<float> output_gradients) const
  {
    return this->backward_helper(layer, std::move(output_gradients), [this](Layer& layer, size_t batch_size, Tensor<float> output_gradients, Tensor<float> input_gradients)
    {
      const size_t size = this->shape.size();

      Tensor<float> outputs        = std::move(layer.saved_tensors[0]);

      auto _outputs          = outputs         .reshape(Shape{batch_size, size});
      auto _input_gradients  = input_gradients .reshape(Shape{batch_size, size});
      auto _output_gradients = output_gradients.reshape(Shape{batch_size, size});

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

