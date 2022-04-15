#include "libkann/layers/SequentialLayer.hpp"
#include <libkann/Random.hpp>

#include <libkann/layers/IdentityLayer.hpp>
#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>
#include <libkann/layers/ConvolutionalLayer.hpp>
#include <libkann/layers/DeconvolutionalLayer.hpp>

#include <libkann/Model.hpp>
#include <libkann/Loader.hpp>
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

int main(int argc, char* argv[])
{
  std::default_random_engine engine(kann::random<std::default_random_engine::result_type>());

  kann::MNISTDataSet trainingDataSet(
    "datasets/mnist/train-images-idx3-ubyte",
    "datasets/mnist/train-labels-idx1-ubyte"
  );

  kann::MNISTDataSet testingDataSet(
    "datasets/mnist/t10k-images-idx3-ubyte",
    "datasets/mnist/t10k-labels-idx1-ubyte"
  );

  if(argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " write/feeedForward/autoencoder [target]" << std::endl;
    return -1;
  }

  std::string subcommand = argv[1];
  if(subcommand == "write")
  {
    // Try to write out the data set
    writeDataSet("output/training", trainingDataSet, kann::MNISTDataSet::COLUMN_IMAGE);
    writeDataSet("output/testing",  testingDataSet,  kann::MNISTDataSet::COLUMN_IMAGE);
  }
  else if(subcommand == "feedforward")
  {
    if(argc != 3)
      return -1;

    // When will we get std::format() support in libstdc++?
    std::stringstream ss;
    ss << "examples/backpropagation/feedforward/" << argv[2] << ".yaml";
    auto filepath = ss.str();

    auto layer = kann::loadLayer(filepath);
    auto model = std::make_shared<kann::Model>(std::move(layer));

    model->randomize(engine);

    auto trainingInputs  = kann::load(trainingDataSet, kann::MNISTDataSet::COLUMN_IMAGE);
    auto trainingOutputs = kann::load(trainingDataSet, kann::MNISTDataSet::COLUMN_LABEL);

    auto testingInputs  = kann::load(testingDataSet, kann::MNISTDataSet::COLUMN_IMAGE);
    auto testingOutputs = kann::load(testingDataSet, kann::MNISTDataSet::COLUMN_LABEL);

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
  else if(subcommand == "autoencoder")
  {
    if(argc != 3)
      return -1;

    std::stringstream ss;

    ss << "examples/backpropagation/autoencoder/" << argv[2] << "-encoder" << ".yaml";
    auto encoderLayer = kann::loadLayer(ss.str());

    ss.str(std::string());

    ss << "examples/backpropagation/autoencoder/" << argv[2] << "-decoder" << ".yaml";
    auto decoderLayer = kann::loadLayer(ss.str());

    auto autoEncoderLayer = std::make_shared<kann::SequentialLayer>();
    autoEncoderLayer->addLayer(encoderLayer, kann::Tag::ENCODER);
    autoEncoderLayer->addLayer(decoderLayer, kann::Tag::DECODER);

    auto decoderModel = std::make_shared<kann::Model>(decoderLayer);
    auto encoderModel = std::make_shared<kann::Model>(encoderLayer);
    auto autoEncoderModel = std::make_shared<kann::Model>(autoEncoderLayer);
    decoderModel->randomize(engine);
    encoderModel->randomize(engine);
    autoEncoderModel->randomize(engine);

    static size_t FEATURES_COUNT = 64;
    const auto reconstructionOutputPath = std::filesystem::path("output/autoencoder-reconstruction");
    const auto outputPath = std::filesystem::path( "output/autoencoder");

    auto trainingData = kann::load(trainingDataSet, kann::MNISTDataSet::COLUMN_IMAGE);
    auto testingData  = kann::load(testingDataSet,  kann::MNISTDataSet::COLUMN_IMAGE); // Not used

    if(!std::filesystem::create_directories(reconstructionOutputPath))
    {
      std::cerr << "Error: Failed to create directories " << reconstructionOutputPath << std::endl;
      return -1;
    }

    if(!std::filesystem::create_directories(outputPath))
    {
      std::cerr << "Error: Failed to create directories " << outputPath << std::endl;
      return -1;
    }

    {
      auto task = kann::train(autoEncoderModel, trainingData, trainingData, LEARNING_RATE, 10);
      while(!task.step())
        kann::displayInfo("Training", task.info());

      std::cout << '\n';
    }


    // Reconstruction
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
    {
      kann::RandomDataSet randomDataSet(FEATURES_COUNT, 500);
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
  else if(subcommand == "gan")
  {
    auto generatorLayer     = kann::loadLayer("examples/backpropagation/gan/generator.yaml");
    auto discriminatorLayer = kann::loadLayer("examples/backpropagation/gan/discriminator.yaml");

    auto GANLayer = std::make_shared<kann::SequentialLayer>();
    GANLayer->addLayer(generatorLayer    , kann::Tag::GAN_GENERATOR);
    GANLayer->addLayer(discriminatorLayer, kann::Tag::GAN_DISCRIMINATOR);

    auto generatorModel = std::make_shared<kann::Model>(generatorLayer);
    auto discriminatorModel = std::make_shared<kann::Model>(discriminatorLayer);
    auto GANModel = std::make_shared<kann::Model>(GANLayer);
    GANModel->randomize(engine);
    generatorModel->randomize(engine);
    discriminatorModel->randomize(engine);

    static constexpr size_t FEATURES_COUNT = 128;

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
    std::cerr << "Usage: " << argv[0] << " write/feeedForward/autoencoder [target]" << std::endl;
    return -1;
  }
}
