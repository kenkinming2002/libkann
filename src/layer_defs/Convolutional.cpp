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
      Variable{.value = MutableTensor::normal(Shape{m_input_channel_count, m_output_channel_count, m_kernel_size.height(), m_kernel_size.width()}, prng, 0.0, 1.0 / (m_kernel_size.width() * m_kernel_size.height())).as_const()}
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
    const Variable& kernels = layer.storage->parameters[0];
    layer.saved_tensors = { inputs };

    const Vec2 padding_size = ((m_output_size - m_input_size) + (m_kernel_size - Vec2(1,1))) / 2;
    return math::cross_correlate2d(inputs, kernels.value, 1, 1, 1, false, false, padding_size);
  }

  Tensor ConvolutionalLayerDef::backward(Layer& layer, Tensor output_gradients) const
  {
    Variable& kernels = layer.storage->parameters[0];
    const Tensor& inputs = layer.saved_tensors[0];

    const Vec2 padding_size = ((m_output_size - m_input_size) + (m_kernel_size - Vec2(1,1))) / 2;
    kernels.gradient = math::cross_correlate2d(inputs, output_gradients, 1, 1, 1, true, false, padding_size);
    return math::convolve2d(output_gradients, kernels.value, 1, 1, 1, false, true, (m_kernel_size - Vec2(1,1)) - padding_size);
  }
}
