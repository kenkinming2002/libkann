#pragma once

#include <libkann/datasets/DataSet.hpp>
#include <libkann/Model.hpp>

namespace kann
{
  LIBKANN_SYMEXPORT void train(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, float learningRate);
  LIBKANN_SYMEXPORT double test(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn);
}
