#include "libkann/ActivationFunction.hpp"
#include "libkann/datasets/DataSet.hpp"
#include <libkann/utilities/random.hpp>

#include <libkann/neural_networks/NeuralNetwork.hpp>
#include <libkann/neural_networks/AutoEncoder.hpp>

#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>
#include <libkann/layers/ConvolutionalLayer.hpp>
#include <libkann/layers/DeconvolutionalLayer.hpp>

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

template<typename NeuralNetwork>
static void attachWeightActivationLayers(NeuralNetwork& nn, const std::vector<size_t>& topology, kann::ActivationFunction::Type activationType)
{
  const auto activationFunction = kann::ActivationFunction(activationType);
  for(size_t i=0; i < topology.size()-1; ++i)
  {
    size_t prevSize = topology[i];
    size_t nextSize = topology[i+1];

    auto weightLayer = std::make_unique<kann::WeightLayer>(prevSize, nextSize);
    auto activationLayer = std::make_unique<kann::ActivationLayer>(weightLayer->outputSize(), activationFunction);
    nn.addLayer(std::move(weightLayer));
    nn.addLayer(std::move(activationLayer));
  }
}

template<typename NeuralNetwork>
static void attachConvolutionActivationLayers(NeuralNetwork& nn, const std::vector<size_t>& topology, size_t& width, size_t& height, size_t kernelSize, kann::ActivationFunction::Type activationType)
{
  const auto activationFunction = kann::ActivationFunction(activationType);
  for(size_t i=0; i < topology.size()-1; ++i)
  {
    size_t prevSize = topology[i];
    size_t nextSize = topology[i+1];

    auto convolutionalLayer = std::make_unique<kann::ConvolutionalLayer>(width, height, kernelSize, prevSize, nextSize);
    auto activationLayer    = std::make_unique<kann::ActivationLayer>(convolutionalLayer->outputSize(), activationFunction);
    nn.addLayer(std::move(convolutionalLayer));
    nn.addLayer(std::move(activationLayer));

    assert(width>kernelSize);
    width -= kernelSize - 1;
    assert(height>kernelSize);
    height -= kernelSize - 1;
  }
}

template<typename NeuralNetwork>
static void attachDeconvolutionActivationLayers(NeuralNetwork& nn, const std::vector<size_t>& topology, size_t& width, size_t& height, size_t kernelSize, kann::ActivationFunction::Type activationType)
{
  const auto activationFunction = kann::ActivationFunction(activationType);
  for(size_t i=0; i < topology.size()-1; ++i)
  {
    size_t prevSize = topology[i];
    size_t nextSize = topology[i+1];

    auto deconvolutionalLayer = std::make_unique<kann::DeconvolutionalLayer>(width, height, kernelSize, prevSize, nextSize);
    auto activationLayer    = std::make_unique<kann::ActivationLayer>(deconvolutionalLayer->outputSize(), activationFunction);
    nn.addLayer(std::move(deconvolutionalLayer));
    nn.addLayer(std::move(activationLayer));

    width += kernelSize - 1;
    height += kernelSize - 1;
  }
}

static void trainAndTestNeuralNetwork(kann::NeuralNetwork& nn, const kann::DataSet& trainingDataSet, const kann::DataSet& testingDataSet)
{
  double correctness;

  correctness = nn.test(testingDataSet);
  std::cout << "Correctness:" << correctness << std::endl;

  nn.train(trainingDataSet, LEARNING_RATE);

  correctness = nn.test(trainingDataSet);
  std::cout << "Training Data Set Correctness:" << correctness << std::endl;

  correctness = nn.test(testingDataSet);
  std::cout << "Testing Data Set Correctness:" << correctness << std::endl;
}

static void trainAndRunAutoEncoder(kann::AutoEncoder& autoEncoder, const kann::DataSet& trainingDataSet, const kann::DataSet& testingDataSet,
    std::filesystem::path reconstructionOutputPath, std::filesystem::path outputPath, size_t featuresCount, size_t generateCount)
{
    Eigen::VectorXd input, output;

    autoEncoder.train(trainingDataSet, LEARNING_RATE);

    // Reconstruction
    if(!std::filesystem::create_directories(reconstructionOutputPath))
    {
      std::cerr << "Error: Failed to create directories " << reconstructionOutputPath << std::endl;
      return;
    }

    for(size_t i = 0; i<trainingDataSet.size(); ++i)
    {
      trainingDataSet.get(i, input, output);
      autoEncoder.feedForward(input);

      std::filesystem::path filepath = reconstructionOutputPath / (std::string("result")+std::to_string(i)+std::string(".bmp"));
      std::ofstream file(filepath, std::ofstream::binary);
      kann::writeImage(file, autoEncoder.output(), kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
    }

    // Generate
    if(!std::filesystem::create_directories(outputPath))
    {
      std::cerr << "Error: Failed to create directories " << outputPath << std::endl;
      return;
    }

    for(size_t i = 0; i<generateCount; ++i)
    {
      const Eigen::VectorXd features = Eigen::VectorXd::Random(featuresCount) * std::sqrt(2.0 / 10);
      autoEncoder.generate(features);

      std::filesystem::path filepath = outputPath / (std::string("result")+std::to_string(i)+std::string(".bmp"));
      std::ofstream file(filepath, std::ofstream::binary);
      kann::writeImage(file, autoEncoder.output(), kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
    }
}

int main(int argc, char* argv[])
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

  if(argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " write/normal/convolution/autoencoder/autoencoder-convolutional" << std::endl;
    return -1;
  }

  std::string subcommand = argv[1];

  if(subcommand == "write")
  {
    // Try to write out the data set
    writeDataSet("output/training", trainingDataSet);
    writeDataSet("output/testing",   testingDataSet);
  }
  else if(subcommand == "normal")
  {
    // Normal Neural Network
    kann::NeuralNetwork nn;
    attachWeightActivationLayers(nn, {kann::MNISTDataSet::IMAGE_SIZE, 30, 30, 30, 10}, kann::ActivationFunction::Type::SIGMOID);
    nn.randomize(engine);
    trainAndTestNeuralNetwork(nn, trainingDataSet, testingDataSet);
  }
  else if(subcommand == "convolution")
  {
    // Convolutional Neural Network
    kann::NeuralNetwork nn;
    size_t width = kann::MNISTDataSet::IMAGE_WIDTH, height = kann::MNISTDataSet::IMAGE_WIDTH;
    attachConvolutionActivationLayers(nn, {1, 3, 3, 1}, width, height, 5, kann::ActivationFunction::Type::SIGMOID);
    attachWeightActivationLayers(nn, {nn.outputSize(), 10}, kann::ActivationFunction::Type::SIGMOID);
    nn.randomize(engine);
    trainAndTestNeuralNetwork(nn, trainingDataSet, testingDataSet);
  }
  else if(subcommand == "autoencoder")
  {
    static constexpr size_t FEATURES_COUNT = 64;

    kann::AutoEncoder autoEncoder;
    attachWeightActivationLayers(autoEncoder, {kann::MNISTDataSet::IMAGE_SIZE, 256, FEATURES_COUNT}, kann::ActivationFunction::Type::SIGMOID);
    autoEncoder.setFeaturesLayer();
    attachWeightActivationLayers(autoEncoder, {FEATURES_COUNT, 256, kann::MNISTDataSet::IMAGE_SIZE}, kann::ActivationFunction::Type::SIGMOID);
    autoEncoder.randomize(engine);

    trainAndRunAutoEncoder(autoEncoder, trainingDataSet, testingDataSet,
      "output/autoencoder-reconstruction", "output/autoencoder",
      FEATURES_COUNT, 500
    );
  }
  else if(subcommand == "autoencoder-convolutional")
  {
    static constexpr size_t FEATURES_COUNT = 64;

    kann::AutoEncoder autoEncoder;
    size_t width = kann::MNISTDataSet::IMAGE_WIDTH, height = kann::MNISTDataSet::IMAGE_WIDTH;

    attachConvolutionActivationLayers(autoEncoder, {1, 5}, width, height, 5, kann::ActivationFunction::Type::SIGMOID);

    const size_t size = autoEncoder.outputSize();
    attachWeightActivationLayers(autoEncoder, {size, FEATURES_COUNT}, kann::ActivationFunction::Type::SIGMOID);
    autoEncoder.setFeaturesLayer();
    attachWeightActivationLayers(autoEncoder, {FEATURES_COUNT, size}, kann::ActivationFunction::Type::SIGMOID);

    attachDeconvolutionActivationLayers(autoEncoder, {5, 1}, width, height, 5, kann::ActivationFunction::Type::SIGMOID);

    attachWeightActivationLayers(autoEncoder, {autoEncoder.outputSize(), autoEncoder.outputSize()}, kann::ActivationFunction::Type::SIGMOID);

    autoEncoder.randomize(engine);

    trainAndRunAutoEncoder(autoEncoder, trainingDataSet, testingDataSet,
      "output/autoencoder-convolutional-reconstruction", "output/autoencoder-convolutional",
      FEATURES_COUNT, 500
    );
  }
  else
  {
    std::cerr << "Error: Invalid Command " << subcommand << std::endl;
    std::cerr << "Usage: " << argv[0] << " write/normal/convolution/autoencoder/autoencoder-convolutional" << std::endl;
    return -1;
  }
}
