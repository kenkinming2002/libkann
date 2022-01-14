#include <libkann/Build.hpp>

#include <libkann/layers/IdentityLayer.hpp>

#include <libkann/FunctionalModel.hpp>

namespace kann
{
  std::shared_ptr<Model> buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers, unsigned tag)
  {
    auto input = OldVariable::constant(layers.front()->inputSize());
    auto output = input;
    for(auto& layer : layers)
      output = output | layer;

    return std::make_shared<FunctionalModel>(std::move(input), std::move(output));
  }

  std::shared_ptr<Model> buildSimpleRecurrentModel(std::vector<std::shared_ptr<Layer>> layers, size_t memory, unsigned tag)
  {
    const size_t inputSize  = layers.front()->inputSize();
    const size_t outputSize = layers.back()->outputSize();

    auto realInput = OldVariable::constant(inputSize-memory);
    auto memoryInput = OldVariable::constant(memory);

    auto input1 = realInput   | std::make_shared<IdentityLayer>(inputSize - memory, inputSize, 0                 );
    auto input2 = memoryInput | std::make_shared<IdentityLayer>(memory            , inputSize, inputSize - memory);
    auto input = input1 + input2;

    auto output = input;
    for(auto& layer : layers)
      output = output | layer;

    auto realOutput   = output | std::make_shared<IdentityLayer>(outputSize, outputSize - memory, 0                  );
    auto memoryOutput = output | std::make_shared<IdentityLayer>(outputSize, memory             , outputSize - memory);

    auto feedBack = FunctionalModel::FeedBack{
      .input = std::move(memoryInput),
      .output = std::move(memoryOutput)
    };
    return std::make_shared<FunctionalModel>(std::move(realInput), std::move(realOutput), std::vector<FunctionalModel::FeedBack>{feedBack});
  }

  std::pair<std::shared_ptr<Model>, std::shared_ptr<Model>> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers)
  {
    auto encoderModel = buildSimpleFeedForwardModel(std::move(encoderLayers));
    encoderModel->tag(TAG_ENCODDER);

    auto decoderModel = buildSimpleFeedForwardModel(std::move(decoderLayers));
    decoderModel->tag(TAG_DECODDER);

    auto input = OldVariable::constant(encoderModel->inputSize());
    auto middle = input | encoderModel;
    auto output = middle | decoderModel;

    auto autoEncoderModel = std::make_shared<FunctionalModel>(std::move(input), std::move(output));

    return {std::move(autoEncoderModel), std::move(decoderModel)};
  }

  std::tuple<std::shared_ptr<Model>, std::shared_ptr<Model>, std::shared_ptr<Model>> buildSimpleGANModel(std::vector<std::shared_ptr<Layer>> generatorLayers, std::vector<std::shared_ptr<Layer>> discriminatorLayers)
  {
    auto generatorModel = buildSimpleFeedForwardModel(std::move(generatorLayers));
    generatorModel->tag(TAG_GAN_GENERATOR);

    auto discriminatorModel = buildSimpleFeedForwardModel(std::move(discriminatorLayers));
    discriminatorModel->tag(TAG_GAN_DISCRIMINATOR);

    auto input = OldVariable::constant(generatorModel->inputSize());
    auto middle = input | generatorModel;
    auto output = middle | discriminatorModel;

    auto GANModel = std::make_shared<FunctionalModel>(std::move(input), std::move(output));

    return {std::move(GANModel), std::move(generatorModel), std::move(discriminatorModel)};
  }
}
