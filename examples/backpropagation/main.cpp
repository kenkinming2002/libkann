#include <libkann/utilities/random.hpp>

#include <libkann/neural_networks/NeuralNetwork.hpp>

#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>
#include <libkann/layers/ConvolutionalLayer.hpp>

#include <libkann/datasets/MNISTDataSet.hpp>
#include <libkann/datasets/write.hpp>

#include <memory>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>

static constexpr double LEARNING_RATE = 0.05;

static void writeDataSet(std::filesystem::path dirpath, const kann::DataSet& dataSet)
{
  if(!std::filesystem::create_directories(dirpath))
  {
    std::cerr << "Error: Failed to create directories " << dirpath << std::endl;
    return;
  }

  Eigen::VectorXd input, output;
  for(size_t i = 0;  i<dataSet.size(); ++i)
  {
    dataSet.get(i, input, output);

    std::filesystem::path filepath = dirpath / (std::string("data")+std::to_string(i)+std::string(".bmp"));
    std::ofstream file(filepath, std::ofstream::binary);
    kann::writeImage(file, input, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
  }
}

int main()
{
  std::default_random_engine engine(random<std::mt19937::result_type>());

  kann::MNISTDataSet trainingDataSet(
    "examples/backpropagation/datasets/train-images-idx3-ubyte",
    "examples/backpropagation/datasets/train-labels-idx1-ubyte"
  );

  kann::MNISTDataSet testingDataSet(
    "examples/backpropagation/datasets/t10k-images-idx3-ubyte",
    "examples/backpropagation/datasets/t10k-labels-idx1-ubyte"
  );

  // Try to write out the data set
  {
    writeDataSet("output/training", trainingDataSet);
    writeDataSet("output/testing",   testingDataSet);
  }

  // Normal Neural Network
  {
    kann::NeuralNetwork nn;

    const size_t topology[] = {kann::MNISTDataSet::IMAGE_SIZE, 30, 30, 30, 10};
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

    double correctness;

    correctness = nn.test(testingDataSet);
    std::cout << "Correctness:" << correctness << std::endl;

    nn.train(trainingDataSet, LEARNING_RATE);

    correctness = nn.test(trainingDataSet);
    std::cout << "Training Data Set Correctness:" << correctness << std::endl;

    correctness = nn.test(testingDataSet);
    std::cout << "Testing Data Set Correctness:" << correctness << std::endl;
  }

  // Convolutional Neural Network
  {
    kann::NeuralNetwork nn;

    const size_t kernelSize = 5;
    const size_t topology[] = {1, 3, 3, 1};
    const auto activationFunction = kann::ActivationFunction(kann::ActivationFunction::Type::SIGMOID);

    size_t width = kann::MNISTDataSet::IMAGE_WIDTH;
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

    auto weightLayer     = std::make_unique<kann::WeightLayer>(nn.outputSize(), 10);
    auto activationLayer = std::make_unique<kann::ActivationLayer>(10, activationFunction);

    nn.addLayer(std::move(weightLayer));
    nn.addLayer(std::move(activationLayer));

    nn.randomize(engine);

    double correctness;

    correctness = nn.test(testingDataSet);
    std::cout << "Correctness:" << correctness << std::endl;

    nn.train(trainingDataSet, LEARNING_RATE);

    correctness = nn.test(trainingDataSet);
    std::cout << "Training Data Set Correctness:" << correctness << std::endl;

    correctness = nn.test(testingDataSet);
    std::cout << "Testing Data Set Correctness:" << correctness << std::endl;
  }
}
