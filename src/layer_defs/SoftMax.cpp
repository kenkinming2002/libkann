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
    node["shape"] = Shape::to_vector(std::static_pointer_cast<const SoftMaxLayerDef>(layer_def)->m_shape);
    return node;
  }

  std::shared_ptr<const LayerDef> SoftMaxLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<SoftMaxLayerDef>();
    layer_def->m_shape = Shape::from_vector(node["shape"].as<std::vector<size_t>>());
    return layer_def;
  }

  std::shared_ptr<LayerStorage> SoftMaxLayerDef::create(std::default_random_engine& prng) const
  {
    return std::make_shared<LayerStorage>();
  }

  Shape SoftMaxLayerDef::input_shape() const
  {
    return m_shape;
  }

  Shape SoftMaxLayerDef::output_shape() const
  {
    return m_shape;
  }

  Tensor SoftMaxLayerDef::forward(Layer& layer, Tensor inputs) const
  {
    const size_t batch_size = inputs.dimension(0);
    const Shape shape       = Shape::concat(Shape(batch_size), m_shape);

    // Exponential map
    MutableTensor outputs = MutableTensor::create(shape);
    math::transform<1>(outputs.as_ref(), {inputs.as_ref()}, [](double /*output*/, double input) { return std::exp(input); });

    // Batch normalization
    MutableTensor factors = MutableTensor::create(Shape(batch_size));
    factors.fill(0.0);
    math::reduce<1>(factors.as_ref(),    {outputs.as_ref().as_const()}, math::Direction::RIGHT, math::ADD);
    math::broadcast<1>(outputs.as_ref(), {factors.as_ref().as_const()}, math::Direction::RIGHT, math::DIV);

    // It is actually better to save outouts
    layer.saved_tensors = { outputs.as_const() };
    return outputs.as_const();
  }

  Tensor SoftMaxLayerDef::backward(Layer& layer, Tensor output_gradients) const
  {
    const size_t batch_size = output_gradients.dimension(0);
    const size_t size       = output_gradients.shape().drop_front(1).size();
    const Shape shape       = Shape::concat(Shape(batch_size), m_shape);

    Tensor outputs                = std::move(layer.saved_tensors[0]);
    MutableTensor input_gradients = MutableTensor::create(shape);
    input_gradients.fill(0.0f);

    for(size_t k=0; k<batch_size; ++k)
    {
      TensorRef        output          = outputs.as_ref()[k].flatten();
      TensorRef        output_gradient = output_gradients.as_ref()[k].flatten();
      MutableTensorRef input_gradient  = input_gradients.as_ref()[k].flatten();
      for(size_t i=0; i<size; ++i)
        for(size_t j=0; j<size; ++j)
          if(i == j)
            input_gradient.get(i) += output_gradient.get(j) * output.get(j) * (1 - output.get(i));
          else
            input_gradient.get(i) -= output_gradient.get(j) * output.get(j) * output.get(i);
    }

    return input_gradients.as_const();
  }
}

