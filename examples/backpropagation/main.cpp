#include <libkann/utilities/random.hpp>

#include <libkann/layers/IdentityLayer.hpp>
#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>
#include <libkann/layers/ConvolutionalLayer.hpp>
#include <libkann/layers/DeconvolutionalLayer.hpp>

#include <libkann/Build.hpp>
#include <libkann/NewModel.hpp>
#include <libkann/Algorithm.hpp>

#include <libkann/datasets/MNISTDataSet.hpp>
#include <libkann/datasets/RandomDataSet.hpp>
#include <libkann/datasets/write.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>

#include <memory>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>

static constexpr double LEARNING_RATE = 0.05;

static void writeDataSet(std::filesystem::path dirpath, const kann::DataSet& dataSet, size_t dataColumn)
{
  if(!std::filesystem::create_directories(dirpath))
  {
    std::cerr << "Error: Failed to create directories " << dirpath << std::endl;
    return;
  }

  Eigen::VectorXd data;
  for(size_t i = 0;  i<dataSet.size(); ++i)
  {
    auto data = dataSet.get(i, dataColumn);

    std::filesystem::path filepath = dirpath / (std::string("data")+std::to_string(i)+std::string(".bmp"));
    auto image = kann::toImage(*data, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
    image.saveToFile(filepath);
  }
}

static void attachWeightActivationLayers(std::vector<std::shared_ptr<kann::NewLayer>>& layers, const std::vector<size_t>& topology, kann::ActivationFunction::Type activationType)
{
  const auto activationFunction = kann::ActivationFunction(activationType);
  for(size_t i=0; i < topology.size()-1; ++i)
  {
    size_t prevSize = topology[i];
    size_t nextSize = topology[i+1];
    layers.push_back(std::make_shared<kann::WeightLayer>(prevSize, nextSize));
    layers.push_back(std::make_shared<kann::ActivationLayer>(nextSize, activationFunction));
  }
}

static void attachConvolutionActivationLayers(std::vector<std::shared_ptr<kann::NewLayer>>& layers, const std::vector<size_t>& topology, size_t& width, size_t& height, size_t kernelSize, kann::ActivationFunction::Type activationType)
{
  const auto activationFunction = kann::ActivationFunction(activationType);
  for(size_t i=0; i < topology.size()-1; ++i)
  {
    size_t prevSize = topology[i];
    size_t nextSize = topology[i+1];

    layers.push_back(std::make_shared<kann::ConvolutionalLayer>(width, height, kernelSize, prevSize, nextSize));
    layers.push_back(std::make_shared<kann::ActivationLayer>(layers.back()->outputSize(), activationFunction));

    assert(width>kernelSize);
    width -= kernelSize - 1;
    assert(height>kernelSize);
    height -= kernelSize - 1;
  }
}

static void attachDeconvolutionActivationLayers(std::vector<std::shared_ptr<kann::NewLayer>>& layers, const std::vector<size_t>& topology, size_t& width, size_t& height, size_t kernelSize, kann::ActivationFunction::Type activationType)
{
  const auto activationFunction = kann::ActivationFunction(activationType);
  for(size_t i=0; i < topology.size()-1; ++i)
  {
    size_t prevSize = topology[i];
    size_t nextSize = topology[i+1];

    layers.push_back(std::make_shared<kann::DeconvolutionalLayer>(width, height, kernelSize, prevSize, nextSize));
    layers.push_back(std::make_shared<kann::ActivationLayer>(layers.back()->outputSize(), activationFunction));

    width += kernelSize - 1;
    height += kernelSize - 1;
  }
}

static void trainAndTestFeedForwardModel(std::shared_ptr<kann::NewModel> model,
    const kann::DataSet& trainingDataSet, const kann::DataSet& testingDataSet,
    size_t inputColumn, size_t outputColumn,
    std::filesystem::path outputPath)
{
  auto trainingInputs  = kann::load(trainingDataSet, inputColumn);
  auto trainingOutputs = kann::load(trainingDataSet, outputColumn);

  auto testingInputs  = kann::load(testingDataSet, inputColumn);
  auto testingOutputs = kann::load(testingDataSet, outputColumn);

  // Testing
  {
    auto task = kann::test(model, testingInputs, testingOutputs);
    while(!task.step())
      kann::displayInfo("Testing", task.info());

    auto correctness = task.get();
    std::cout << "correctness:" << correctness << std::endl;
  }

  // Training
  {
    auto train = kann::train(model, trainingInputs, trainingOutputs, LEARNING_RATE, 10);
    while(!train.step())
      kann::displayInfo("Training", train.info());

    std::cout << '\n';
  }

  // Testing
  {
    auto task = kann::test(model, testingInputs, testingOutputs);
    while(!task.step())
      kann::displayInfo("Testing", task.info());

    auto correctness = task.get();
    std::cout << "correctness:" << correctness << std::endl;
  }
}

static void trainAndRunAutoEncoder(std::shared_ptr<kann::NewModel> autoEncoderModel, std::shared_ptr<kann::NewModel> decoderModel,
    const kann::DataSet& trainingDataSet, const kann::DataSet& testingDataSet, size_t dataColumn,
    std::filesystem::path reconstructionOutputPath, std::filesystem::path outputPath, size_t featuresCount, size_t generateCount)
{
  auto trainingData = kann::load(trainingDataSet, dataColumn);
  auto testingData  = kann::load(testingDataSet, dataColumn); // Not used

  {
    auto task = kann::train(autoEncoderModel, trainingData, trainingData, LEARNING_RATE, 10);
    while(!task.step())
      kann::displayInfo("Training", task.info());

    std::cout << '\n';
  }

  // Reconstruction
  if(!std::filesystem::create_directories(reconstructionOutputPath))
  {
    std::cerr << "Error: Failed to create directories " << reconstructionOutputPath << std::endl;
    return;
  }

  {
    auto task = kann::run(autoEncoderModel, trainingData);
    while(!task.step())
    {
      auto info = task.info();
      std::filesystem::path filepath = reconstructionOutputPath / (std::string("result")+std::to_string(info.i)+std::string(".bmp"));
      auto image = kann::toImage(*info.output, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
      image.saveToFile(filepath);
    }
  }

  // Generate
  if(!std::filesystem::create_directories(outputPath))
  {
    std::cerr << "Error: Failed to create directories " << outputPath << std::endl;
    return;
  }

  {
    kann::RandomDataSet randomDataSet(featuresCount, generateCount);
    auto randomData = kann::load(randomDataSet, kann::RandomDataSet::COLUMN_DATA);

    auto task = kann::run(decoderModel, randomData);
    while(!task.step())
    {
      auto info = task.info();
      std::filesystem::path filepath = outputPath / (std::string("result")+std::to_string(info.i)+std::string(".bmp"));
      auto image = kann::toImage(*info.output, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
      image.saveToFile(filepath);
    }

  }
}

int main(int argc, char* argv[])
{
  std::default_random_engine engine(random<std::mt19937::result_type>());

  kann::MNISTDataSet trainingDataSet(
    "datasets/mnist/train-images-idx3-ubyte",
    "datasets/mnist/train-labels-idx1-ubyte"
  );

  kann::MNISTDataSet testingDataSet(
    "datasets/mnist/t10k-images-idx3-ubyte",
    "datasets/mnist/t10k-labels-idx1-ubyte"
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
    writeDataSet("output/training", trainingDataSet, kann::MNISTDataSet::COLUMN_IMAGE);
    writeDataSet("output/testing",  testingDataSet,  kann::MNISTDataSet::COLUMN_IMAGE);
  }
  else if(subcommand == "serialize")
  {
    {
      std::vector<std::shared_ptr<kann::NewLayer>> layers;
      layers.push_back(std::make_shared<kann::IdentityLayer>(kann::MNISTDataSet::IMAGE_SIZE, kann::MNISTDataSet::IMAGE_SIZE, 0));
      attachWeightActivationLayers(layers, {kann::MNISTDataSet::IMAGE_SIZE, 30, 30, 30, 10}, kann::ActivationFunction::Type::SIGMOID);
      layers.push_back(std::make_shared<kann::IdentityLayer>(10, 10, 0));

      auto model = kann::buildSimpleFeedForwardModel(std::move(layers));
      model->randomize();

      std::ofstream file("output/model1.json");
      cereal::JSONOutputArchive archive(file);
      archive(model);
    }

    {
      std::shared_ptr<kann::NewModel> model;
      {
        std::ifstream file("output/model1.json");
        cereal::JSONInputArchive archive(file);
        archive(model);
      }

      {
        std::ofstream file("output/model2.json");
        cereal::JSONOutputArchive archive(file);
        archive(model);
      }
    }
  }
  else if(subcommand == "normal")
  {
    // Normal Neural Network
    std::vector<std::shared_ptr<kann::NewLayer>> layers;
    layers.push_back(std::make_shared<kann::IdentityLayer>(kann::MNISTDataSet::IMAGE_SIZE, kann::MNISTDataSet::IMAGE_SIZE, 0));
    attachWeightActivationLayers(layers, {kann::MNISTDataSet::IMAGE_SIZE, 30, 30, 30, 10}, kann::ActivationFunction::Type::SIGMOID);
    layers.push_back(std::make_shared<kann::IdentityLayer>(10, 10, 0));

    auto model = kann::buildSimpleFeedForwardModel(std::move(layers));
    model->randomize();

    trainAndTestFeedForwardModel(model, trainingDataSet, testingDataSet, kann::MNISTDataSet::COLUMN_IMAGE, kann::MNISTDataSet::COLUMN_LABEL, "output/normal.dot");
  }
  else if(subcommand == "convolution")
  {
    std::vector<std::shared_ptr<kann::NewLayer>> layers;
    size_t width = kann::MNISTDataSet::IMAGE_WIDTH, height = kann::MNISTDataSet::IMAGE_WIDTH;
    attachConvolutionActivationLayers(layers, {1, 3, 3, 1}, width, height, 5, kann::ActivationFunction::Type::SIGMOID);
    attachWeightActivationLayers(layers, {layers.back()->outputSize(), 10}, kann::ActivationFunction::Type::SIGMOID);

    auto model = kann::buildSimpleFeedForwardModel(std::move(layers));
    model->randomize();

    trainAndTestFeedForwardModel(model, trainingDataSet, testingDataSet, kann::MNISTDataSet::COLUMN_IMAGE, kann::MNISTDataSet::COLUMN_LABEL, "output/convolution.dot");
  }
  else if(subcommand == "recurrent")
  {
    // Normal Neural Network
    const size_t MEMORY = 20;

    std::vector<std::shared_ptr<kann::NewLayer>> layers;
    attachWeightActivationLayers(layers, {kann::MNISTDataSet::IMAGE_SIZE+MEMORY, 30, 30, 30, 10+MEMORY}, kann::ActivationFunction::Type::SIGMOID);

    auto model = kann::buildSimpleRecurrentModel(std::move(layers), MEMORY);
    model->randomize();

    trainAndTestFeedForwardModel(model, trainingDataSet, testingDataSet, kann::MNISTDataSet::COLUMN_IMAGE, kann::MNISTDataSet::COLUMN_LABEL, "output/recurrent.dot");
  }
  else if(subcommand == "autoencoder")
  {
    static constexpr size_t FEATURES_COUNT = 64;

    std::vector<std::shared_ptr<kann::NewLayer>> encoderLayers;
    attachWeightActivationLayers(encoderLayers, {kann::MNISTDataSet::IMAGE_SIZE, 256, FEATURES_COUNT}, kann::ActivationFunction::Type::SIGMOID);

    std::vector<std::shared_ptr<kann::NewLayer>> decoderLayers;
    attachWeightActivationLayers(decoderLayers, {FEATURES_COUNT, 256, kann::MNISTDataSet::IMAGE_SIZE}, kann::ActivationFunction::Type::SIGMOID);

    auto [autoEncoderModel, decoderModel] = kann::buildSimpleAutoEncoderModel(std::move(encoderLayers), std::move(decoderLayers));
    autoEncoderModel->randomize();
    decoderModel->randomize();

    trainAndRunAutoEncoder(autoEncoderModel, decoderModel, trainingDataSet, testingDataSet, kann::MNISTDataSet::COLUMN_IMAGE,
      "output/autoencoder-reconstruction", "output/autoencoder",
      FEATURES_COUNT, 500
    );
  }
  else if(subcommand == "autoencoder-convolutional")
  {
    static constexpr size_t FEATURES_COUNT = 64;

    size_t width = kann::MNISTDataSet::IMAGE_WIDTH, height = kann::MNISTDataSet::IMAGE_WIDTH;
    std::vector<std::shared_ptr<kann::NewLayer>> encoderLayers;

    attachConvolutionActivationLayers(encoderLayers, {1, 5}, width, height, 5, kann::ActivationFunction::Type::SIGMOID);
    const auto size = encoderLayers.back()->outputSize();
    attachWeightActivationLayers(encoderLayers, {size, FEATURES_COUNT}, kann::ActivationFunction::Type::SIGMOID);

    std::vector<std::shared_ptr<kann::NewLayer>> decoderLayers;
    attachWeightActivationLayers(decoderLayers, {FEATURES_COUNT, size}, kann::ActivationFunction::Type::SIGMOID);
    attachDeconvolutionActivationLayers(decoderLayers, {5, 1}, width, height, 5, kann::ActivationFunction::Type::SIGMOID);

    attachWeightActivationLayers(decoderLayers, {decoderLayers.back()->outputSize(), decoderLayers.back()->outputSize()}, kann::ActivationFunction::Type::SIGMOID);

    auto [autoEncoderModel, decoderModel] = kann::buildSimpleAutoEncoderModel(std::move(encoderLayers), std::move(decoderLayers));
    autoEncoderModel->randomize();
    decoderModel->randomize();

    trainAndRunAutoEncoder(autoEncoderModel, decoderModel, trainingDataSet, testingDataSet, kann::MNISTDataSet::COLUMN_IMAGE,
      "output/autoencoder-convolutional-reconstruction", "output/autoencoder-convolutional",
      FEATURES_COUNT, 500
    );
  }
  else if(subcommand == "gan")
  {
    static constexpr size_t FEATURES_COUNT = 128;

    std::vector<std::shared_ptr<kann::NewLayer>> generatorLayers;
    attachWeightActivationLayers(generatorLayers, {FEATURES_COUNT, 256, kann::MNISTDataSet::IMAGE_SIZE}, kann::ActivationFunction::Type::SIGMOID);

    std::vector<std::shared_ptr<kann::NewLayer>> discriminatorLayers;
    attachWeightActivationLayers(discriminatorLayers, {kann::MNISTDataSet::IMAGE_SIZE, 512, 128, 1}, kann::ActivationFunction::Type::SIGMOID);


    auto [GANModel, generatorModel, discriminatorModel] = kann::buildSimpleGANModel(std::move(generatorLayers), std::move(discriminatorLayers));
    GANModel->randomize();
    generatorModel->randomize();
    discriminatorModel->randomize();

    std::filesystem::path outputDirectory("output/gan");
    std::filesystem::create_directories(outputDirectory / "Training");
    std::filesystem::create_directories(outputDirectory / "Output");

    {
      kann::RandomDataSet latentDataSet(FEATURES_COUNT, trainingDataSet.size());

      auto trainingData = kann::load(trainingDataSet, kann::MNISTDataSet::COLUMN_IMAGE);
      auto latentData   = kann::load(latentDataSet, kann::RandomDataSet::COLUMN_DATA);

      auto task = kann::trainGAN(GANModel, generatorModel, discriminatorModel, trainingData, latentData, LEARNING_RATE, 10);
      while(!task.step())
      {
        auto info = task.info();

        std::ostringstream fileName;
        fileName << std::setfill('0') << std::setw(std::ceil(std::log10(info.size))) << info.i << ".png";
        auto image = kann::toImage(*info.generatorOutput, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
        image.saveToFile(outputDirectory / "Training" / fileName.str());

        kann::displayInfo("Training", info);
      }
      std::cout << '\n';
    }

    {
      kann::RandomDataSet latentDataSet(FEATURES_COUNT, 1000);
      auto latentData = kann::load(latentDataSet, kann::RandomDataSet::COLUMN_DATA);

      auto task = kann::run(generatorModel, latentData);
      while(!task.step())
      {
        auto info = task.info();

        std::ostringstream fileName;
        fileName << std::setfill('0') << std::setw(std::ceil(std::log10(info.size))) << info.i << ".png";
        auto image = kann::toImage(*info.output, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
        image.saveToFile(outputDirectory / "Output" / fileName.str());

        kann::displayInfo("Generating", info);
      }
      std::cout << '\n';
    }
  }
  else
  {
    std::cerr << "Error: Invalid Command " << subcommand << std::endl;
    std::cerr << "Usage: " << argv[0] << " write/normal/convolution/autoencoder/autoencoder-convolutional" << std::endl;
    return -1;
  }
}
