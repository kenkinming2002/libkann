#pragma once

#include <libkann/datasets/DataSet.hpp>
#include <libkann/Model.hpp>
#include <libkann/Tensor.hpp>

#include <libkann/Task.hpp>

#include <functional>

namespace kann
{
  // FIXME: Replace callbacks with something like coroutine

  struct Info
  {
    const std::shared_ptr<const Model> model;

    size_t i;
    size_t size;

    std::shared_ptr<const Tensor> input;
    std::shared_ptr<const Tensor> output;
    std::shared_ptr<const Tensor> expectedOutput;

    double cost;
  };
  typedef std::function<bool(Info)> Callback;

  struct GANInfo
  {
    const std::shared_ptr<Model> GANModel;
    const std::shared_ptr<Model> generatorModel;
    const std::shared_ptr<Model> discriminatorModel;

    size_t i;
    size_t size;

    std::shared_ptr<const Tensor> GANOutput;
    std::shared_ptr<const Tensor> generatorOutput;
    std::shared_ptr<const Tensor> discriminatorOutput;
  };
  typedef std::function<bool(GANInfo)> GANCallback;

  Callback defaultCallback(std::string_view name);
  GANCallback defaultGANCallback(std::string_view name);

  // New API
  LIBKANN_SYMEXPORT std::vector<std::shared_ptr<const Tensor>> load(const DataSet& dataSet, size_t column);

  LIBKANN_SYMEXPORT void run(std::shared_ptr<const Model> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      Callback callback);

  LIBKANN_SYMEXPORT void train(std::shared_ptr<Model> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs,
      double learningRate, size_t batchSize,
      Callback callback = defaultCallback("Training"));

  LIBKANN_SYMEXPORT void trainGAN(std::shared_ptr<Model> model,
      std::shared_ptr<Model> generatorModel,
      std::shared_ptr<Model> discriminatorModel,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> latentInputs,
      double learningRate, size_t batchSize,
      GANCallback callback = defaultGANCallback("Training"));

  LIBKANN_SYMEXPORT double test(std::shared_ptr<const Model> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs,
      Callback callback = defaultCallback("Testing"));

}
