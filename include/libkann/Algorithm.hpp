#pragma once

#include <libkann/datasets/DataSet.hpp>
#include <libkann/Model.hpp>

#include <functional>

namespace kann
{
  // We must pass the model explicitly because copy could have been made
  struct Info
  {
    const Model& model;

    size_t i;
    size_t size;

    Eigen::VectorXd output;
    Eigen::VectorXd expectedOutput;

    double cost;
  };
  typedef std::function<bool(Info)> Callback;

  struct GANInfo
  {
    const Model& GANModel;
    const Model& generatorModel;
    const Model& discriminatorModel;

    size_t i;
    size_t size;

    Eigen::VectorXd GANOutput;
    Eigen::VectorXd generatorOutput;
    Eigen::VectorXd discriminatorOutput;
  };
  typedef std::function<bool(GANInfo)> GANCallback;

  Callback defaultCallback(std::string_view name);
  GANCallback defaultGANCallback(std::string_view name);

  LIBKANN_SYMEXPORT void run(Model& model, const DataSet& dataSet, size_t column, Callback callback);

  LIBKANN_SYMEXPORT void train(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, float learningRate, Callback callback = defaultCallback("Training"));
  LIBKANN_SYMEXPORT double test(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, Callback callback = defaultCallback("Testing"));

  LIBKANN_SYMEXPORT void trainGAN(Model& GANModel, Model& generatorModel, Model& discriminatorModel,
      const DataSet& dataSetLatent, const DataSet& dataSet,
      size_t columnLatent, size_t column,
      float learningRate, GANCallback callback = defaultGANCallback("Training"));
}
