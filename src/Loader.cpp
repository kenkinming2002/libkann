#include <libkann/Loader.hpp>

#include <libkann/layer_defs/Sequential.hpp>
#include <libkann/layer_defs/Recurrent.hpp>
#include <libkann/layer_defs/Identity.hpp>
#include <libkann/layer_defs/Activation.hpp>
#include <libkann/layer_defs/Weight.hpp>
#include <libkann/layer_defs/Convolutional.hpp>
#include <libkann/layer_defs/Deconvolutional.hpp>

#include <yaml-cpp/yaml.h>

#include <range/v3/all.hpp>

namespace kann
{
  static layer_def_t load_layer_def(YAML::Node root)
  {
    std::shared_ptr<LayerDef> result;

    auto type = root["type"].as<std::string>();
    if(type == "sequential")
    {
      auto layers = root["layers"];

      result = std::make_shared<SequentialLayerDef>();
      result->sub_layer_defs = layers
        | ranges::views::transform([&](YAML::Node child) { return load_layer_def(child); })
        | ranges::to_vector;

      return result;
    }
    else if(type == "recurrent")
    {
      auto memory_size = root["memory"].as<size_t>();
      auto layers = root["layers"];

      result = std::make_shared<RecurrentLayerDef>(memory_size);
      result->sub_layer_defs = layers
        | ranges::views::transform([&](YAML::Node child) { return load_layer_def(child); })
        | ranges::to_vector;

      return result;
    }
    else if(type == "identity")
    {
      auto input_size  = root["input_size"].as<size_t>();
      auto output_size = root["output_size"].as<size_t>();
      auto offset      = root["offset"].as<size_t>();
      result = std::make_shared<IdentityLayerDef>(input_size, output_size, offset);
    }
    else if(type == "activation")
    {
      size_t size = root["size"].as<size_t>();

      auto function = root["function"].as<std::string>();
      ActivationFunction::Type function_type;
      if(function == "identity")
        function_type = ActivationFunction::Type::IDENTITY;
      else if(function == "sigmoid")
        function_type = ActivationFunction::Type::SIGMOID;
      else if(function == "tanh")
        function_type = ActivationFunction::Type::TANH;
      else
        throw std::runtime_error("Unrecognized activation layer function:" + function);

      result = std::make_shared<ActivationLayerDef>(size, kann::ActivationFunction(function_type));
    }
    else if(type == "weight")
    {
      size_t input_size  = root["input_size"].as<size_t>();
      size_t output_size = root["output_size"].as<size_t>();
      result = std::make_shared<WeightLayerDef>(input_size, output_size);
    }
    else if(type == "convolution")
    {
      size_t input_width  = root["input_width"].as<size_t>();
      size_t input_height = root["input_height"].as<size_t>();
      size_t kernel_size  = root["kernel_size"].as<size_t>();
      size_t input_channel_count  = root["input_channel_count"].as<size_t>();
      size_t output_channel_count = root["output_channel_count"].as<size_t>();
      result = std::make_shared<ConvolutionalLayerDef>(input_width, input_height, kernel_size, input_channel_count, output_channel_count);
    }
    else if(type == "deconvolution")
    {
      size_t input_width  = root["input_width"].as<size_t>();
      size_t input_height = root["input_height"].as<size_t>();
      size_t kernel_size  = root["kernel_size"].as<size_t>();
      size_t input_channel_count  = root["input_channel_count"].as<size_t>();
      size_t output_channel_count = root["output_channel_count"].as<size_t>();
      result = std::make_shared<DeconvolutionalLayerDef>(input_width, input_height, kernel_size, input_channel_count, output_channel_count);
    }
    else
      throw std::runtime_error("Unrecognized layer type:" + type);

    result->tag = Tag::ALL;
    return result;
  }

  layer_def_t load_layer_def(const std::string& filename)
  {
    YAML::Node root = YAML::LoadFile(filename);
    return load_layer_def(root);
  }

  layer_def_t load_layer_def(std::istream& is)
  {
    YAML::Node root = YAML::Load(is);
    return load_layer_def(root);
  }
}
