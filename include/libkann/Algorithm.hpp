#pragma once

#include <libkann/datasets/DataSet.hpp>
#include <libkann/FunctionalModel.hpp>

#include <functional>

namespace kann
{
  // We must pass the model explicitly because copy could have been made
  struct Info
  {
    const FunctionalModel& model;

    size_t i;
    size_t size;

    Eigen::VectorXd output;
    Eigen::VectorXd expectedOutput;

    double cost;
  };
  typedef std::function<bool(Info)> Callback;

  struct GANInfo
  {
    const FunctionalModel& GANModel;
    const FunctionalModel& generatorModel;
    const FunctionalModel& discriminatorModel;

    size_t i;
    size_t size;

    Eigen::VectorXd GANOutput;
    Eigen::VectorXd generatorOutput;
    Eigen::VectorXd discriminatorOutput;
  };
  typedef std::function<bool(GANInfo)> GANCallback;

  Callback defaultCallback(std::string_view name);
  GANCallback defaultGANCallback(std::string_view name);

  LIBKANN_SYMEXPORT void run(FunctionalModel& model, const DataSet& dataSet, size_t column, Callback callback);

  LIBKANN_SYMEXPORT void train(FunctionalModel& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, float learningRate, Callback callback = defaultCallback("Training"));
  LIBKANN_SYMEXPORT double test(FunctionalModel& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, Callback callback = defaultCallback("Testing"));

  LIBKANN_SYMEXPORT void trainGAN(FunctionalModel& GANModel, FunctionalModel& generatorModel, FunctionalModel& discriminatorModel,
      const DataSet& dataSetLatent, const DataSet& dataSet,
      size_t columnLatent, size_t column,
      float learningRate, GANCallback callback = defaultGANCallback("Training"));
}
