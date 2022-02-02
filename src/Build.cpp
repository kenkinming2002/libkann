#include <libkann/Build.hpp>

#include <libkann/layers/IdentityLayer.hpp>
#include <libkann/layers/SequentialLayer.hpp>
#include <libkann/layers/RecurrentLayer.hpp>

#include <libkann/NewModel.hpp>

namespace kann
{
  std::shared_ptr<NewLayer> buildSimpleFeedForwardLayer(std::vector<std::shared_ptr<NewLayer>> layers)
  {
    auto result = std::make_shared<SequentialLayer>();
    for(auto& layer : layers)
      result->addLayer(std::move(layer));

    return result;
  }

  std::shared_ptr<NewLayer> buildSimpleRecurrentLayer(std::vector<std::shared_ptr<NewLayer>> layers, size_t memory)
  {
    auto result = std::make_shared<RecurrentLayer>(memory);
    for(auto& layer : layers)
      result->addLayer(std::move(layer));

    return result;
  }

  std::shared_ptr<NewModel> buildSimpleFeedForwardModel(std::vector<std::shared_ptr<NewLayer>> layers)
  {
    auto resultLayer = buildSimpleFeedForwardLayer(std::move(layers));
    return std::make_shared<NewModel>(std::move(resultLayer));
  }

  std::shared_ptr<NewModel> buildSimpleRecurrentModel(std::vector<std::shared_ptr<NewLayer>> layers, size_t memory)
  {
    auto resultLayer = buildSimpleRecurrentLayer(std::move(layers), memory);
    return std::make_shared<NewModel>(std::move(resultLayer));
  }

  std::pair<std::shared_ptr<NewModel>, std::shared_ptr<NewModel>> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<NewLayer>> encoderLayers, std::vector<std::shared_ptr<NewLayer>> decoderLayers)
  {
    auto encoderLayer = buildSimpleFeedForwardLayer(std::move(encoderLayers));
    encoderLayer->tag(NEW_TAG_ENCODDER);

    auto decoderLayer = buildSimpleFeedForwardLayer(std::move(decoderLayers));
    decoderLayer->tag(NEW_TAG_DECODDER);

    auto autoEncoderLayer = buildSimpleFeedForwardLayer({encoderLayer, decoderLayer});

    return {
      std::make_shared<NewModel>(std::move(autoEncoderLayer)),
      std::make_shared<NewModel>(std::move(decoderLayer))
    };
  }

  std::tuple<std::shared_ptr<NewModel>, std::shared_ptr<NewModel>, std::shared_ptr<NewModel>> buildSimpleGANModel(std::vector<std::shared_ptr<NewLayer>> generatorLayers, std::vector<std::shared_ptr<NewLayer>> discriminatorLayers)
  {
    auto generatorLayer = buildSimpleFeedForwardLayer(std::move(generatorLayers));
    generatorLayer->tag(NEW_TAG_GAN_GENERATOR);

    auto discriminatorLayer = buildSimpleFeedForwardLayer(std::move(discriminatorLayers));
    discriminatorLayer->tag(NEW_TAG_GAN_DISCRIMINATOR);

    auto GANLayer = buildSimpleFeedForwardLayer({generatorLayer, discriminatorLayer});

    return {
      std::make_shared<NewModel>(std::move(GANLayer)),
      std::make_shared<NewModel>(std::move(generatorLayer)),
      std::make_shared<NewModel>(std::move(discriminatorLayer))
    };
  }
}
