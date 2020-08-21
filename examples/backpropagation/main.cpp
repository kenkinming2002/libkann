#include "DataSet.hpp"

#include <libkann/utilities/random.hpp>
#include <libkann/NeuralNetwork.hpp>

#include <random>
#include <chrono>

static constexpr double LEARNING_RATE = 0.05;

int main()
{
  std::mt19937 generator(random<std::mt19937::result_type>());
  NeuralNetwork nn({DataSet::INPUT_LAYER_SIZE, 30, 30, 30, DataSet::OUTPUT_LAYER_SIZE}, generator, ActivationFunction(ActivationFunction::Type::SIGMOID));

  DataSet testingDataSet(
    "examples/backpropagation/datasets/t10k-images-idx3-ubyte", 
    "examples/backpropagation/datasets/t10k-labels-idx1-ubyte"
  );
  DataSet trainingDataSet(
    "examples/backpropagation/datasets/train-images-idx3-ubyte", 
    "examples/backpropagation/datasets/train-labels-idx1-ubyte"
  );

  testingDataSet.test(nn);
  trainingDataSet.train(nn, LEARNING_RATE);
  testingDataSet.test(nn);
}
