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
  else if(subcommand == "convolution")
  {
    // Convolutional Neural Network
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
  else if(subcommand == "autoencoder")
  {
    static constexpr size_t FEATURES_COUNT = 64;

    kann::AutoEncoder autoEncoder;

    const size_t topology[] = {kann::MNISTDataSet::IMAGE_SIZE, 256, FEATURES_COUNT, 256, kann::MNISTDataSet::IMAGE_SIZE};
    const auto activationFunction = kann::ActivationFunction(kann::ActivationFunction::Type::SIGMOID);

    for(size_t i=0; i < sizeof topology / sizeof topology[0] - 1; ++i)
    {
      size_t prevSize = topology[i];
      size_t nextSize = topology[i+1];

      auto weightLayer = std::make_unique<kann::WeightLayer>(prevSize, nextSize);
      auto activationLayer = std::make_unique<kann::ActivationLayer>(nextSize, activationFunction);

      autoEncoder.addLayer(std::move(weightLayer));
      autoEncoder.addLayer(std::move(activationLayer));

      if(nextSize == FEATURES_COUNT)
        autoEncoder.setFeaturesLayer();
    }

    autoEncoder.randomize(engine);
    autoEncoder.train(trainingDataSet, LEARNING_RATE);

    // Reconstruction
    {
      const std::filesystem::path dirpath("output/autoencoder-reconstruction");
      if(!std::filesystem::create_directories(dirpath))
      {
        std::cerr << "Error: Failed to create directories " << dirpath << std::endl;
        return -1;
      }

      Eigen::VectorXd input, output;
      for(size_t i = 0; i<trainingDataSet.size(); ++i)
      {
        trainingDataSet.get(i, input, output);
        autoEncoder.feedForward(input);

        std::filesystem::path filepath = dirpath / (std::string("result")+std::to_string(i)+std::string(".bmp"));
        std::ofstream file(filepath, std::ofstream::binary);
        kann::writeImage(file, autoEncoder.output(), kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
      }
    }

    // Generate
    {
      static constexpr size_t GENERATE_COUNT = 100;

      const std::filesystem::path dirpath("output/autoencoder");
      if(!std::filesystem::create_directories(dirpath))
      {
        std::cerr << "Error: Failed to create directories " << dirpath << std::endl;
        return -1;
      }

      for(size_t i = 0; i<GENERATE_COUNT; ++i)
      {
        const Eigen::VectorXd features = Eigen::VectorXd::Random(FEATURES_COUNT) * std::sqrt(2.0 / 10);
        autoEncoder.generate(features);

        std::filesystem::path filepath = dirpath / (std::string("result")+std::to_string(i)+std::string(".bmp"));
        std::ofstream file(filepath, std::ofstream::binary);
        kann::writeImage(file, autoEncoder.output(), kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
      }
    }
  }
  else if(subcommand == "autoencoder-convolutional")
  {
    static constexpr size_t FEATURES_COUNT = 64;

    kann::AutoEncoder autoEncoder;

    const size_t kernelSize = 5;
    const auto activationFunction = kann::ActivationFunction(kann::ActivationFunction::Type::SIGMOID);

    size_t width = kann::MNISTDataSet::IMAGE_WIDTH;

    // Convolutional Layer
    {
      const size_t topology[] = {1, 5};

      for(size_t i=0; i < sizeof topology / sizeof topology[0] - 1; ++i)
      {
        size_t prevSize = topology[i];
        size_t nextSize = topology[i+1];

        auto convolutionalLayer = std::make_unique<kann::ConvolutionalLayer>(width, width, kernelSize, prevSize, nextSize);
        auto activationLayer    = std::make_unique<kann::ActivationLayer>(convolutionalLayer->outputSize(), activationFunction);

        autoEncoder.addLayer(std::move(convolutionalLayer));
        autoEncoder.addLayer(std::move(activationLayer));

        assert(width>kernelSize);
        width -= kernelSize - 1;
      }

    }

    const size_t size = autoEncoder.outputSize();

    auto prevWeightLayer     = std::make_unique<kann::WeightLayer>(size, FEATURES_COUNT);
    auto prevActivationLayer = std::make_unique<kann::ActivationLayer>(FEATURES_COUNT, activationFunction);
    autoEncoder.addLayer(std::move(prevWeightLayer));
    autoEncoder.addLayer(std::move(prevActivationLayer));

    autoEncoder.setFeaturesLayer();

    auto nextWeightLayer     = std::make_unique<kann::WeightLayer>(FEATURES_COUNT, size);
    auto nextActivationLayer = std::make_unique<kann::ActivationLayer>(size, activationFunction);
    autoEncoder.addLayer(std::move(nextWeightLayer));
    autoEncoder.addLayer(std::move(nextActivationLayer));

    // Deconvolutional Layer
    {
      const size_t topology[] = {5, 1};

      for(size_t i=0; i < sizeof topology / sizeof topology[0] - 1; ++i)
      {
        size_t prevSize = topology[i];
        size_t nextSize = topology[i+1];

        auto deconvolutionalLayer = std::make_unique<kann::DeconvolutionalLayer>(width, width, kernelSize, prevSize, nextSize);
        auto activationLayer    = std::make_unique<kann::ActivationLayer>(deconvolutionalLayer->outputSize(), activationFunction);

        autoEncoder.addLayer(std::move(deconvolutionalLayer));
        autoEncoder.addLayer(std::move(activationLayer));

        width += kernelSize - 1;
      }
    }

    // Note: YOu must add a dense layer (or in our library called
    //       kann::WeightLayer) at the end of any convolutional neural network.
    //       They are the layer that actually do the heavy lifting of
    //       classification.
    auto lastWeightLayer     = std::make_unique<kann::WeightLayer>(autoEncoder.outputSize(), autoEncoder.outputSize());
    auto lastActivationLayer = std::make_unique<kann::ActivationLayer>(autoEncoder.outputSize(), activationFunction);
    autoEncoder.addLayer(std::move(lastWeightLayer));
    autoEncoder.addLayer(std::move(lastActivationLayer));

    autoEncoder.randomize(engine);
    autoEncoder.train(trainingDataSet, LEARNING_RATE);

    // Reconstruction
    {
      const std::filesystem::path dirpath("output/autoencoder-convolutional-reconstruction");
      if(!std::filesystem::create_directories(dirpath))
      {
        std::cerr << "Error: Failed to create directories " << dirpath << std::endl;
        return -1;
      }

      Eigen::VectorXd input, output;
      for(size_t i = 0; i<trainingDataSet.size(); ++i)
      {
        trainingDataSet.get(i, input, output);
        autoEncoder.feedForward(input);

        std::filesystem::path filepath = dirpath / (std::string("result")+std::to_string(i)+std::string(".bmp"));
        std::ofstream file(filepath, std::ofstream::binary);
        kann::writeImage(file, autoEncoder.output(), kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
      }
    }

    // Generate
    {
      static constexpr size_t GENERATE_COUNT = 100;

      const std::filesystem::path dirpath("output/autoencoder-convolutional");
      if(!std::filesystem::create_directories(dirpath))
      {
        std::cerr << "Error: Failed to create directories " << dirpath << std::endl;
        return -1;
      }

      for(size_t i = 0; i<GENERATE_COUNT; ++i)
      {
        const Eigen::VectorXd features = Eigen::VectorXd::Random(FEATURES_COUNT) * std::sqrt(2.0 / 10);
        autoEncoder.generate(features);

        std::filesystem::path filepath = dirpath / (std::string("result")+std::to_string(i)+std::string(".bmp"));
        std::ofstream file(filepath, std::ofstream::binary);
        kann::writeImage(file, autoEncoder.output(), kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
      }
    }
  }
  else
  {
    std::cerr << "Error: Invalid Command " << subcommand << std::endl;
    std::cerr << "Usage: " << argv[0] << " write/normal/convolution/autoencoder/autoencoder-convolutional" << std::endl;
    return -1;
  }
}
