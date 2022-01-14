#pragma once

#include <libkann/Model.hpp>

namespace kann
{
  std::shared_ptr<Model> buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers, unsigned tag = TAG_DEFAULT);
  std::shared_ptr<Model> buildSimpleRecurrentModel(std::vector<std::shared_ptr<Layer>> layers, size_t memory, unsigned tag = TAG_DEFAULT);

  /* The returned auto encoder model is used for training purposes
   * whereas random data can be feed into the decoder model to obtain output.
   *
   * Since Model hold std::shared_ptr to Layer, training using the auto encoder
   * model could be reflected in the decoder model.
   *
   * @return [auto encoder model, decoder model] */
  std::pair<std::shared_ptr<Model>, std::shared_ptr<Model>> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers);

  /* The returned GAN and discriminator model is used for training purpose.
   *
   * Since Model holds std::shared_ptr to Layer, the training result could be
   * reflected in the generator model.
   *
   * @return [GAN Model, generator model, discriminator model] */
  std::tuple<std::shared_ptr<Model>, std::shared_ptr<Model>, std::shared_ptr<Model>> buildSimpleGANModel(std::vector<std::shared_ptr<Layer>> generatorLayers, std::vector<std::shared_ptr<Layer>> discriminatorLayers);
}
