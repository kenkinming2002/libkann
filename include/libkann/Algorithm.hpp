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

    const Eigen::VectorXd& input;
    const Eigen::VectorXd& output;
    const Eigen::VectorXd& expectedOutput;

    double cost;
  };
  typedef std::function<void(Info)> Callback;

  Callback defaultCallback(std::string_view name);

  LIBKANN_SYMEXPORT void train(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, float learningRate, Callback callback = defaultCallback("Training"));
  LIBKANN_SYMEXPORT double test(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, Callback callback = defaultCallback("Testing"));
}
