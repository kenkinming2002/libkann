#pragma once

#include <libkann/datasets/DataSet.hpp>
#include <libkann/Model.hpp>
#include <libkann/Tensor.hpp>

#include <libkann/Task.hpp>

#include <functional>

namespace kann
{
  struct Info
  {
    std::shared_ptr<const Model> model;

    size_t i;
    size_t size;

    std::shared_ptr<const Tensor> input;
    std::shared_ptr<const Tensor> output;
    std::shared_ptr<const Tensor> expectedOutput;

    double cost;
  };

  struct GANInfo
  {
    std::shared_ptr<Model> GANModel;
    std::shared_ptr<Model> generatorModel;
    std::shared_ptr<Model> discriminatorModel;

    size_t i;
    size_t size;

    std::shared_ptr<const Tensor> GANOutput;
    std::shared_ptr<const Tensor> generatorOutput;
    std::shared_ptr<const Tensor> discriminatorOutput;
  };

  LIBKANN_SYMEXPORT void displayInfo(std::string_view name, const Info& info);
  LIBKANN_SYMEXPORT void displayInfo(std::string_view name, const GANInfo& info);

  // New API
  LIBKANN_SYMEXPORT std::vector<std::shared_ptr<const Tensor>> load(const DataSet& dataSet, size_t column);

  LIBKANN_SYMEXPORT Task<void, Info> run(std::shared_ptr<const Model> model,
      std::vector<std::shared_ptr<const Tensor>> inputs);

  LIBKANN_SYMEXPORT Task<void, Info> train(std::shared_ptr<Model> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs,
      double learningRate, size_t batchSize);

  LIBKANN_SYMEXPORT Task<void, GANInfo> trainGAN(std::shared_ptr<Model> model,
      std::shared_ptr<Model> generatorModel,
      std::shared_ptr<Model> discriminatorModel,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> latentInputs,
      double learningRate, size_t batchSize);

  LIBKANN_SYMEXPORT Task<double, Info> test(std::shared_ptr<const Model> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs);

}
