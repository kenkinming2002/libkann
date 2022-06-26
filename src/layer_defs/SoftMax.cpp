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
    return this->forward_helper(layer, std::move(inputs), [](Layer& layer, size_t batch_size, Tensor<float> inputs, Tensor<float> outputs)
    {
      math::transform<1>(outputs.as_ref(), {inputs.as_const_ref()}, [](double /*output*/, double input) { return std::exp(input); });

      // Batch normalization
      Tensor<float> factors = Tensor<float>::create(Shape(batch_size));
      factors.as_ref().fill(0.0);
      math::reduce<1>(factors.as_ref(),    {outputs.as_const_ref()}, math::Direction::RIGHT, math::ADD);
      math::broadcast<1>(outputs.as_ref(), {factors.as_const_ref()}, math::Direction::RIGHT, math::DIV);

      // It is actually better to save outouts
      layer.saved_tensors.clear();
      layer.saved_tensors.reserve(1);
      layer.saved_tensors.push_back(outputs.clone());

      return outputs;
    });
  }

  Tensor<float> SoftMaxLayerDef::backward(Layer& layer, Tensor<float> output_gradients) const
  {
    return this->backward_helper(layer, std::move(output_gradients), [](Layer& layer, size_t batch_size, Tensor<float> output_gradients, Tensor<float> input_gradients)
    {
      Tensor<float> outputs        = std::move(layer.saved_tensors[0]);
      input_gradients.as_ref().fill(0.0f);
      for(size_t k=0; k<batch_size; ++k)
      {
        TensorRef<const float> output          = outputs.as_const_ref()[k].flatten();
        TensorRef<const float> output_gradient = output_gradients.as_const_ref()[k].flatten();
        TensorRef<float>       input_gradient  = input_gradients.as_ref()[k].flatten();

        const size_t size = input_gradient.size();
        for(size_t i=0; i<size; ++i)
          for(size_t j=0; j<size; ++j)
            if(i == j)
              input_gradient[i].as_scalar() += output_gradient[j].as_scalar() * output[j].as_scalar() * (1 - output[i].as_scalar());
            else
              input_gradient[i].as_scalar() -= output_gradient[j].as_scalar() * output[j].as_scalar() * output[i].as_scalar();
      }

      return input_gradients;
    });
  }
}

