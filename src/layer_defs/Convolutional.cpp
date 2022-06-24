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
    node["input_channel_count"]  = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->input_channel_count;
    node["output_channel_count"] = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->output_channel_count;
    node["input_width"]          = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->input_size.width();
    node["input_height"]         = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->input_size.height();
    node["output_width"]         = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->output_size.width();
    node["output_height"]        = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->output_size.height();
    node["kernel_width"]         = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->kernel_size.width();
    node["kernel_height"]        = std::static_pointer_cast<const ConvolutionalLayerDef>(layer_def)->kernel_size.height();
    return node;
  }

  std::shared_ptr<const LayerDef> ConvolutionalLayerDef::load(YAML::Node node)
  {
    auto layer_def = std::make_shared<ConvolutionalLayerDef>();
    layer_def->input_channel_count  = node["input_channel_count"].as<size_t>();
    layer_def->output_channel_count = node["output_channel_count"].as<size_t>();
    layer_def->input_size           = Vec2(node["input_width"].as<size_t>(),  node["input_height"].as<size_t>());
    layer_def->output_size          = Vec2(node["output_width"].as<size_t>(), node["output_height"].as<size_t>());
    layer_def->kernel_size          = Vec2(node["kernel_width"].as<size_t>(), node["kernel_height"].as<size_t>());
    return layer_def;
  }

  std::shared_ptr<LayerStorage> ConvolutionalLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer_storage = std::make_shared<LayerStorage>();

    Tensor<float> kernels = Tensor<float>::create(Shape{input_channel_count, output_channel_count, kernel_size.height(), kernel_size.width()});
    kernels.as_ref().fill_normal(prng, 0.0, 1.0 / (kernel_size.width() * kernel_size.height()));

    layer_storage->parameters.reserve(1);
    layer_storage->parameters.push_back(Variable{.value = std::move(kernels)});

    return layer_storage;
  }

  Shape ConvolutionalLayerDef::get_input_shape() const
  {
    return Shape{input_channel_count, input_size.height(), input_size.width()};
  }

  Shape ConvolutionalLayerDef::get_output_shape() const
  {
    return Shape{output_channel_count, output_size.height(), output_size.width()};
  }

  Tensor<float> ConvolutionalLayerDef::forward(Layer& layer, Tensor<float> inputs) const
  {
    const size_t batch_size = inputs.as_ref().dimension(0);
    const Variable& kernels = layer.storage->parameters[0];

    Tensor<float> outputs = Tensor<float>::create(Shape::concat(Shape(batch_size), this->get_output_shape()));
    math::image2d_operation(outputs.as_ref(), inputs.as_const_ref(), false, kernels.value.as_const_ref(), false, math::Image2DOperation::CROSS_CORRELATION);

    layer.saved_tensors.clear();
    layer.saved_tensors.reserve(1);
    layer.saved_tensors.push_back(std::move(inputs));
    return outputs;
  }

  Tensor<float> ConvolutionalLayerDef::backward(Layer& layer, Tensor<float> output_gradients) const
  {
    Variable& kernels = layer.storage->parameters[0];
    const Tensor<float>& inputs = layer.saved_tensors[0];

    Tensor<float> input_gradients  = Tensor<float>::create(inputs.as_ref().shape());
    Tensor<float> kernel_gradients = Tensor<float>::create(kernels.value.as_ref().shape());
    math::image2d_operation(input_gradients.as_ref(),  output_gradients.as_const_ref(), false, kernels.value.as_const_ref(),    true,  math::Image2DOperation::CONVOLUTION);
    math::image2d_operation(kernel_gradients.as_ref(), inputs.as_const_ref(),           true,  output_gradients.as_const_ref(), false, math::Image2DOperation::CROSS_CORRELATION);

    kernels.gradient = std::move(kernel_gradients);
    return input_gradients;
  }
}
