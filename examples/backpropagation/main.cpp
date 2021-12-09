#include "DataSet.hpp"
#include "libkann/ActivationFunction.hpp"

#include <libkann/utilities/random.hpp>

#include <libkann/neural_networks/NeuralNetwork.hpp>

#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>
#include <libkann/layers/ConvolutionalLayer.hpp>

#include <memory>
#include <random>
#include <chrono>

static constexpr double LEARNING_RATE = 0.05;

int main()
{
  std::default_random_engine engine(random<std::mt19937::result_type>());

  DataSet testingDataSet(
    "examples/backpropagation/datasets/t10k-images-idx3-ubyte",
    "examples/backpropagation/datasets/t10k-labels-idx1-ubyte"
  );

  DataSet trainingDataSet(
    "examples/backpropagation/datasets/train-images-idx3-ubyte",
    "examples/backpropagation/datasets/train-labels-idx1-ubyte"
  );


  // Normal Neural Network
  {
    kann::NeuralNetwork nn;

    const size_t topology[] = {DataSet::INPUT_LAYER_SIZE, 30, 30, 30, DataSet::OUTPUT_LAYER_SIZE};
    const auto activationFunction = kann::ActivationFunction(kann::ActivationFunction::Type::SIGMOID);

    for(size_t i=0; i < sizeof topology / sizeof topology[0] - 1; ++i)
    {
      size_t prevSize = topology[i];
      size_t nextSize = topology[i+1];
      auto weightLayer = std::make_unique<kann::WeightLayer>(prevSize, nextSize);

      auto activationLayer = std::make_unique<kann::ActivationLayer>(nextSize, activationFunction);
      nn.addLayer(std::move(weightLayer));
      nn.addLayer(std::move(activationLayer));
    }

    nn.randomize(engine);

    testingDataSet.test(nn);
    trainingDataSet.train(nn, LEARNING_RATE);
    testingDataSet.test(nn);
  }

  // Convolutional Neural Network
  {
    kann::NeuralNetwork nn;

    const size_t kernelSize = 5;
    const size_t topology[] = {1, 3, 3, 1};
    const auto activationFunction = kann::ActivationFunction(kann::ActivationFunction::Type::SIGMOID);

    size_t width = Data::IMAGE_WIDTH;
    for(size_t i=0; i < sizeof topology / sizeof topology[0] - 1; ++i)
    {
      size_t prevSize = topology[i];
      size_t nextSize = topology[i+1];

      auto convolutionalLayer = std::make_unique<kann::ConvolutionalLayer>(width, width, kernelSize, prevSize, nextSize);

      auto activationLayer    = std::make_unique<kann::ActivationLayer>(convolutionalLayer->outputSize(), activationFunction);

      nn.addLayer(std::move(convolutionalLayer));
      nn.addLayer(std::move(activationLayer));

      assert(width>kernelSize);
      width -= kernelSize - 1;
    }

    auto weightLayer     = std::make_unique<kann::WeightLayer>(nn.outputSize(), DataSet::OUTPUT_LAYER_SIZE);
    auto activationLayer = std::make_unique<kann::ActivationLayer>(DataSet::OUTPUT_LAYER_SIZE, activationFunction);

    nn.addLayer(std::move(weightLayer));
    nn.addLayer(std::move(activationLayer));

    nn.randomize(engine);

    testingDataSet.test(nn);
    trainingDataSet.train(nn, LEARNING_RATE);
    testingDataSet.test(nn);
  }
}
