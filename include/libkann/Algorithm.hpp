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
    CRef<Model> model;

    size_t i;
    size_t size;

    CRef<Tensor> input;
    CRef<Tensor> output;
    CRef<Tensor> expectedOutput;

    double cost;
  };

  struct GANInfo
  {
    Ref<Model> GANModel;
    Ref<Model> generatorModel;
    Ref<Model> discriminatorModel;

    size_t i;
    size_t size;

    CRef<Tensor> GANOutput;
    CRef<Tensor> generatorOutput;
    CRef<Tensor> discriminatorOutput;
  };

  void displayInfo(std::string_view name, const Info& info);
  void displayInfo(std::string_view name, const GANInfo& info);

  // New API
  std::vector<CRef<Tensor>> load(const DataSet& dataSet, size_t column);

  Task<void, Info> run(Ref<Model> model,
      std::vector<CRef<Tensor>> inputs);

  Task<void, Info> train(Ref<Model> model,
      std::vector<CRef<Tensor>> inputs,
      std::vector<CRef<Tensor>> expectedOutputs,
      double learningRate, size_t batchSize);

  Task<void, GANInfo> trainGAN(Ref<Model> model,
      Ref<Model> generatorModel,
      Ref<Model> discriminatorModel,
      std::vector<CRef<Tensor>> inputs,
      std::vector<CRef<Tensor>> latentInputs,
      double learningRate, size_t batchSize);

  Task<double, Info> test(Ref<Model> model,
      std::vector<CRef<Tensor>> inputs,
      std::vector<CRef<Tensor>> expectedOutputs);

}
