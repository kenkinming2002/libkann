#include <libkann/Build.hpp>

#include <libkann/layers/IdentityLayer.hpp>
#include <libkann/layers/SequentialLayer.hpp>
#include <libkann/layers/RecurrentLayer.hpp>

#include <libkann/Model.hpp>

namespace kann
{
  std::shared_ptr<Layer> buildSimpleFeedForwardLayer(std::vector<std::shared_ptr<Layer>> layers)
  {
    auto result = std::make_shared<SequentialLayer>();
    for(auto& layer : layers)
      result->addLayer(std::move(layer));

    return result;
  }

  std::shared_ptr<Layer> buildSimpleRecurrentLayer(std::vector<std::shared_ptr<Layer>> layers, size_t memory)
  {
    auto result = std::make_shared<RecurrentLayer>(memory);
    for(auto& layer : layers)
      result->addLayer(std::move(layer));

    return result;
  }

  std::shared_ptr<Model> buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers)
  {
    auto resultLayer = buildSimpleFeedForwardLayer(std::move(layers));
    return std::make_shared<Model>(std::move(resultLayer));
  }

  std::shared_ptr<Model> buildSimpleRecurrentModel(std::vector<std::shared_ptr<Layer>> layers, size_t memory)
  {
    auto resultLayer = buildSimpleRecurrentLayer(std::move(layers), memory);
    return std::make_shared<Model>(std::move(resultLayer));
  }

  std::pair<std::shared_ptr<Model>, std::shared_ptr<Model>> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers)
  {
    auto encoderLayer = buildSimpleFeedForwardLayer(std::move(encoderLayers));
    encoderLayer->tag(TAG_ENCODER);

    auto decoderLayer = buildSimpleFeedForwardLayer(std::move(decoderLayers));
    decoderLayer->tag(TAG_DECODER);

    auto autoEncoderLayer = buildSimpleFeedForwardLayer({encoderLayer, decoderLayer});

    return {
      std::make_shared<Model>(std::move(autoEncoderLayer)),
      std::make_shared<Model>(std::move(decoderLayer))
    };
  }

  std::tuple<std::shared_ptr<Model>, std::shared_ptr<Model>, std::shared_ptr<Model>> buildSimpleGANModel(std::vector<std::shared_ptr<Layer>> generatorLayers, std::vector<std::shared_ptr<Layer>> discriminatorLayers)
  {
    auto generatorLayer = buildSimpleFeedForwardLayer(std::move(generatorLayers));
    generatorLayer->tag(TAG_GAN_GENERATOR);

    auto discriminatorLayer = buildSimpleFeedForwardLayer(std::move(discriminatorLayers));
    discriminatorLayer->tag(TAG_GAN_DISCRIMINATOR);

    auto GANLayer = buildSimpleFeedForwardLayer({generatorLayer, discriminatorLayer});

    return {
      std::make_shared<Model>(std::move(GANLayer)),
      std::make_shared<Model>(std::move(generatorLayer)),
      std::make_shared<Model>(std::move(discriminatorLayer))
    };
  }
}
