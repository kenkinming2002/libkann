#pragma once

#include <libkann/datasets/DataSet.hpp>
#include <libkann/Model.hpp>
#include <libkann/Tensor.hpp>

#include <functional>

namespace kann
{
  // We must pass the model explicitly because copy could have been made
  struct Info
  {
    const std::shared_ptr<Model> model;

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

  LIBKANN_SYMEXPORT void run(std::shared_ptr<Model> model, const DataSet& dataSet, size_t column, Callback callback);

  LIBKANN_SYMEXPORT void train(std::shared_ptr<Model> model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, float learningRate, size_t batchSize = 1, Callback callback = defaultCallback("Training"));
  LIBKANN_SYMEXPORT double test(std::shared_ptr<Model> model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, Callback callback = defaultCallback("Testing"));

  LIBKANN_SYMEXPORT void trainGAN(std::shared_ptr<Model> GANModel, std::shared_ptr<Model> generatorModel, std::shared_ptr<Model> discriminatorModel,
      const DataSet& dataSetLatent, const DataSet& dataSet,
      size_t columnLatent, size_t column,
      float learningRate, GANCallback callback = defaultGANCallback("Training"));
}
