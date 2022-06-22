#include <libkann/layer_defs/Convolutional.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Layer.hpp>
#include <libkann/LayerStorage.hpp>

#include <libkann/Math.hpp>

namespace kann
{
  YAML::Node ConvolutionalLayerDef::save(std::shared_ptr<const LayerDef> layer_def)
  {
    YAML::Node node;
    node["input_channel_count"]  = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_input_channel_count;
    node["output_channel_count"] = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_output_channel_count;
    node["input_width"]          = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_input_size.width();
    node["input_height"]         = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_input_size.height();
    node["output_width"]         = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_output_size.width();
    node["output_height"]        = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_output_size.height();
    node["kernel_width"]         = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_kernel_size.width();
    node["kernel_height"]        = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->m_kernel_size.height();
    return node;
  }

  std::shared_ptr<const LayerDef> ConvolutionalLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<ConvolutionalLayerDef>();
    layer_def->m_input_channel_count  = node["input_channel_count"].as<size_t>();
    layer_def->m_output_channel_count = node["output_channel_count"].as<size_t>();
    layer_def->m_input_size           = Vec2(node["input_width"].as<size_t>(),  node["input_height"].as<size_t>());
    layer_def->m_output_size          = Vec2(node["output_width"].as<size_t>(), node["output_height"].as<size_t>());
    layer_def->m_kernel_size          = Vec2(node["kernel_width"].as<size_t>(), node["kernel_height"].as<size_t>());
    return layer_def;
  }

  std::shared_ptr<LayerStorage> ConvolutionalLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer_storage = std::make_shared<LayerStorage>();
    layer_storage->parameters = {
      Variable{.value = MutableTensor::normal(Shape{m_input_channel_count, m_output_channel_count, m_kernel_size.height(), m_kernel_size.width()}, prng, 0.0, 1.0 / (m_kernel_size.width() * m_kernel_size.height()))}
    };
    return layer_storage;
  }

  Shape ConvolutionalLayerDef::input_shape() const
  {
    return Shape{m_input_channel_count, m_input_size.height(), m_input_size.width()};
  }

  Shape ConvolutionalLayerDef::output_shape() const
  {
    return Shape{m_output_channel_count, m_output_size.height(), m_output_size.width()};
  }

  Tensor ConvolutionalLayerDef::forward(Layer& layer, Tensor inputs) const
  {
    const size_t batch_size = inputs.dimension(0);

    const Variable& kernels = layer.storage->parameters[0];
    layer.saved_tensors = { inputs };

    MutableTensor outputs = MutableTensor::create(Shape::concat(Shape(batch_size), this->output_shape()));
    math::image2d_operation(outputs.as_ref(), inputs.as_ref(), false, kernels.value.as_ref().as_const(), false, math::Image2DOperation::CROSS_CORRELATION);
    return outputs.as_const();
  }

  Tensor ConvolutionalLayerDef::backward(Layer& layer, Tensor output_gradients) const
  {
    Variable& kernels = layer.storage->parameters[0];
    const Tensor& inputs = layer.saved_tensors[0];

    MutableTensor input_gradients  = MutableTensor::create(inputs.shape());
    MutableTensor kernel_gradients = MutableTensor::create(kernels.value.shape());
    math::image2d_operation(input_gradients.as_ref(),  output_gradients.as_ref(), false, kernels.value.as_ref().as_const(), true,  math::Image2DOperation::CONVOLUTION);
    math::image2d_operation(kernel_gradients.as_ref(), inputs.as_ref(),           true,  output_gradients.as_ref(),         false, math::Image2DOperation::CROSS_CORRELATION);

    kernels.gradient = std::move(kernel_gradients).as_const();
    return std::move(input_gradients).as_const();
  }
}
