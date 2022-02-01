#pragma once

#include <libkann/NewModel.hpp>

namespace kann
{
  std::shared_ptr<NewModel> buildSimpleFeedForwardModel(std::vector<std::shared_ptr<NewLayer>> layers);
  std::shared_ptr<NewModel> buildSimpleRecurrentModel(std::vector<std::shared_ptr<NewLayer>> layers, size_t memory);

  /* The returned auto encoder model is used for training purposes
   * whereas random data can be feed into the decoder model to obtain output.
   *
   * Since Model hold std::shared_ptr to Layer, training using the auto encoder
   * model could be reflected in the decoder model.
   *
   * @return [auto encoder model, decoder model] */
  std::pair<std::shared_ptr<NewModel>, std::shared_ptr<NewModel>> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<NewLayer>> encoderLayers, std::vector<std::shared_ptr<NewLayer>> decoderLayers);

  /* The returned GAN and discriminator model is used for training purpose.
   *
   * Since Model holds std::shared_ptr to Layer, the training result could be
   * reflected in the generator model.
   *
   * @return [GAN Model, generator model, discriminator model] */
  std::tuple<std::shared_ptr<NewModel>, std::shared_ptr<NewModel>, std::shared_ptr<NewModel>> buildSimpleGANModel(std::vector<std::shared_ptr<NewLayer>> generatorLayers, std::vector<std::shared_ptr<NewLayer>> discriminatorLayers);
}
