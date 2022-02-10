#include <libkann/Loader.hpp>

#include <libkann/layers/SequentialLayer.hpp>
#include <libkann/layers/RecurrentLayer.hpp>
#include <libkann/layers/IdentityLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>
#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ConvolutionalLayer.hpp>
#include <libkann/layers/DeconvolutionalLayer.hpp>

#include <charconv>

#include <yaml-cpp/yaml.h>

namespace kann
{
  static std::shared_ptr<Layer> loadLayer(YAML::Node root)
  {
    auto type = root["type"].as<std::string>();
    if(type == "sequential")
    {
      auto result = std::make_shared<SequentialLayer>();
      auto layers = root["layers"];
      for(auto child : layers)
        result->addLayer(loadLayer(child)); // TODO: Support tagging

      return result;
    }
    else if(type == "recurrent")
    {
      auto memory = root["memory"].as<size_t>();
      auto result = std::make_shared<RecurrentLayer>(memory);
      auto layers = root["layers"];
      for(auto child : layers)
        result->addLayer(loadLayer(child)); // TODO: Support tagging

      return result;
    }
    else if(type == "identity")
    {
      auto inputSize  = root["input_size"].as<size_t>();
      auto outputSize = root["output_size"].as<size_t>();
      auto offset     = root["offset"].as<size_t>();
      return std::make_shared<IdentityLayer>(inputSize, outputSize, offset);
    }
    else if(type == "activation")
    {
      size_t size = root["size"].as<size_t>();

      auto function = root["function"].as<std::string>();
      ActivationFunction::Type functionType;
      if(function == "identity")
        functionType = ActivationFunction::Type::IDENTITY;
      else if(function == "sigmoid")
        functionType = ActivationFunction::Type::SIGMOID;
      else if(function == "tanh")
        functionType = ActivationFunction::Type::TANH;
      else
        throw std::runtime_error("Unrecognized activation layer function:" + function);

      return std::make_shared<ActivationLayer>(size, kann::ActivationFunction(functionType));
    }
    else if(type == "weight")
    {
      size_t inputSize  = root["input_size"].as<size_t>();
      size_t outputSize = root["output_size"].as<size_t>();
      return std::make_shared<WeightLayer>(inputSize, outputSize);
    }
    else if(type == "convolution")
    {
      size_t inputWidth  = root["input_width"].as<size_t>();
      size_t inputHeight = root["input_height"].as<size_t>();
      size_t kernelSize  = root["kernel_size"].as<size_t>();
      size_t inputChanelCount  = root["input_channel_count"].as<size_t>();
      size_t outputChanelCount = root["output_channel_count"].as<size_t>();
      return std::make_shared<ConvolutionalLayer>(inputWidth, inputHeight, kernelSize, inputChanelCount, outputChanelCount);
    }
    else if(type == "deconvolution")
    {
      size_t inputWidth  = root["input_width"].as<size_t>();
      size_t inputHeight = root["input_height"].as<size_t>();
      size_t kernelSize  = root["kernel_size"].as<size_t>();
      size_t inputChanelCount  = root["input_channel_count"].as<size_t>();
      size_t outputChanelCount = root["output_channel_count"].as<size_t>();
      return std::make_shared<DeconvolutionalLayer>(inputWidth, inputHeight, kernelSize, inputChanelCount, outputChanelCount);
    }
    else
      throw std::runtime_error("Unrecognized layer type:" + type);
  }

  std::shared_ptr<Layer> loadLayer(const std::string& filename)
  {
    YAML::Node root = YAML::LoadFile(filename);
    return loadLayer(root);
  }

  std::shared_ptr<Layer> loadLayer(std::istream& is)
  {
    YAML::Node root = YAML::Load(is);
    return loadLayer(root);
  }
}
