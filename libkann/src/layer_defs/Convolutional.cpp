#include <libkann/layer_defs/Convolutional.hpp>

#include <libtensor/Tensor.hpp>
#include <libtensor/Initializer.hpp>
#include <libkann/Layer.hpp>

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

  std::shared_ptr<Layer> ConvolutionalLayerDef::create() const
  {
    auto layer = std::make_shared<ConvolutionalLayer>();
    layer->def = *this;
    return layer;
  }

  const LayerDef& ConvolutionalLayer::get_def() const { return def; }

  tensor::Shape ConvolutionalLayer::get_input_shape()  const { return tensor::Shape::make(def.input_channel_count,  def.input_size.height(),  def.input_size.width()); }
  tensor::Shape ConvolutionalLayer::get_output_shape() const { return tensor::Shape::make(def.output_channel_count, def.output_size.height(), def.output_size.width()); }

  void ConvolutionalLayer::initialize(std::default_random_engine& prng)
  {
    this->kernels = Variable::create_normal(
        tensor::Shape::make(def.input_channel_count, def.output_channel_count, def.kernel_size.height(), def.kernel_size.width()),
        0.0, 1.0 / (def.kernel_size.width() * def.kernel_size.height()), prng);
  }

  std::unordered_map<std::string, const Variable*> ConvolutionalLayer::parameters_map() const { return {{"kernels", &kernels}}; }
  std::unordered_map<std::string, Variable*>       ConvolutionalLayer::parameters_map()       { return {{"kernels", &kernels}}; }

  tensor::Tensor<float> ConvolutionalLayer::forward(tensor::Tensor<float> inputs)
  {
    saved_tensors = { inputs };
    auto kernels = this->kernels.value;
    auto outputs = tensor::image2d_cross_correlate<float>(inputs, false, kernels, false, def.output_size);
    return outputs;
  }

  tensor::Tensor<float> ConvolutionalLayer::backward(tensor::Tensor<float> output_gradients)
  {
    auto inputs = saved_tensors[0];
    auto kernels = this->kernels.value;
    auto input_gradients  = tensor::image2d_convolve       <float>(output_gradients, false, kernels, true, def.input_size);
    auto kernel_gradients = tensor::image2d_cross_correlate<float>(inputs, true, output_gradients, false,  def.kernel_size);
    this->kernels.gradient = kernel_gradients;
    return input_gradients;
  }
}
