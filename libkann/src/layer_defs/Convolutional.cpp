#include <libkann/layer_defs/Convolutional.hpp>

#include <libtensor/Tensor.hpp>
#include <libtensor/Initializer.hpp>
#include <libkann/Layer.hpp>
#include <libkann/LayerStorage.hpp>

#include <libtensor/Image.hpp>

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
    layer_def->input_size           = tensor::Vec2(node["input_width"].as<size_t>(),  node["input_height"].as<size_t>());
    layer_def->output_size          = tensor::Vec2(node["output_width"].as<size_t>(), node["output_height"].as<size_t>());
    layer_def->kernel_size          = tensor::Vec2(node["kernel_width"].as<size_t>(), node["kernel_height"].as<size_t>());
    return layer_def;
  }

  std::shared_ptr<LayerStorage> ConvolutionalLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer_storage = std::make_shared<LayerStorage>();

    Variable kernels = Variable::create_normal(tensor::Shape::make(input_channel_count, output_channel_count, kernel_size.height(), kernel_size.width()), 0.0, 1.0 / (kernel_size.width() * kernel_size.height()), prng);
    layer_storage->parameters.reserve(1);
    layer_storage->parameters.push_back(std::move(kernels));

    return layer_storage;
  }

  tensor::Shape ConvolutionalLayerDef::get_input_shape() const
  {
    return tensor::Shape::make(input_channel_count, input_size.height(), input_size.width());
  }

  tensor::Shape ConvolutionalLayerDef::get_output_shape() const
  {
    return tensor::Shape::make(output_channel_count, output_size.height(), output_size.width());
  }

  tensor::Tensor<const float> ConvolutionalLayerDef::forward(Layer& layer, tensor::Tensor<const float> inputs) const
  {
    layer.saved_tensors.clear();
    layer.saved_tensors.reserve(1);
    layer.saved_tensors.push_back(inputs);

    auto kernels = layer.storage->parameters[0].value;

    auto outputs = tensor::image2d_cross_correlate<float>(inputs, false, kernels, false, output_size);
    return outputs;
  }

  tensor::Tensor<const float> ConvolutionalLayerDef::backward(Layer& layer, tensor::Tensor<const float> output_gradients) const
  {
    auto inputs = layer.saved_tensors[0];
    auto kernels = layer.storage->parameters[0].value;

    auto input_gradients  = tensor::image2d_convolve       <float>(output_gradients, false, kernels, true, input_size);
    auto kernel_gradients = tensor::image2d_cross_correlate<float>(inputs, true, output_gradients, false,  kernel_size);

    layer.storage->parameters[0].gradient = kernel_gradients;
    return input_gradients;
  }
}
