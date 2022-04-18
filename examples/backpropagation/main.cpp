#include <libkann/Random.hpp>

#include <libkann/layers/IdentityLayer.hpp>
#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>
#include <libkann/layers/ConvolutionalLayer.hpp>
#include <libkann/layers/DeconvolutionalLayer.hpp>
#include <libkann/layers/SequentialLayer.hpp>

#include <libkann/Model.hpp>
#include <libkann/Loader.hpp>
#include <libkann/Algorithm.hpp>

#include <libkann/optimizers/AdamOptimizer.hpp>
#include <libkann/optimizers/SimpleOptimizer.hpp>

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

static constexpr double LEARNING_RATE  = 0.05;
static constexpr size_t BATCH_SIZE     = 10;
static constexpr size_t FEATURES_COUNT = 64;

/* Note:
 *
 * Auto encoder and Adam Optimizer does not play nicely together - at
 * least when used in conjuction with sum of square difference cost function.
 * For better performance, use Simple Optimizer with auto encoder. */
static const auto OPTIMIZER      = std::make_shared<kann::AdamOptimizer>(0.001, 0.9, 0.999, 1e-10);
//static const auto OPTIMIZER      = std::make_shared<kann::SimpleOptimizer>(0.05);

int main(int argc, char* argv[])
{
  std::default_random_engine engine(kann::random<std::default_random_engine::result_type>());

  kann::MNISTDataSet training_dataset(
    "datasets/mnist/train-images-idx3-ubyte",
    "datasets/mnist/train-labels-idx1-ubyte"
  );

  kann::MNISTDataSet testing_dataset(
    "datasets/mnist/t10k-images-idx3-ubyte",
    "datasets/mnist/t10k-labels-idx1-ubyte"
  );

  auto training_inputs  = kann::load(training_dataset, kann::MNISTDataSet::COLUMN_IMAGE);
  auto training_outputs = kann::load(training_dataset, kann::MNISTDataSet::COLUMN_LABEL);

  auto testing_inputs  = kann::load(testing_dataset, kann::MNISTDataSet::COLUMN_IMAGE);
  auto testing_outputs = kann::load(testing_dataset, kann::MNISTDataSet::COLUMN_LABEL);


  if(argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " write/feeedForward/autoencoder [target]" << std::endl;
    return -1;
  }

  std::string subcommand = argv[1];
  if(subcommand == "write")
  {
    for(const auto& [output_path, data] : {std::pair{"output/training", training_inputs}, std::pair{"output/testing", testing_inputs}})
    {
      std::filesystem::create_directories(output_path);
      for(size_t i=0; i<data.size(); ++i)
        kann::toImage(*data[i], kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH)
          .saveToFile(std::string(output_path) + "/data" + std::to_string(i) + ".bmp");
    }
  }
  else if(subcommand == "feedforward")
  {
    if(argc != 3)
      return -1;

    auto layer = kann::loadLayer("examples/backpropagation/feedforward/" + std::string(argv[2]) + ".yaml");
    auto model = std::make_shared<kann::Model>(std::move(layer));

    model->randomize(engine);
    model->compile(BATCH_SIZE, OPTIMIZER, {kann::Tag::ALL});

    // Testing
    {
      auto task = kann::test(model, testing_inputs, testing_outputs);
      while(!task.step())
        kann::displayInfo("Testing", task.info());

      std::cout << "correctness:" << task.get() << std::endl;
    }

    // Training
    {
      auto train = kann::train(model, training_inputs, training_outputs, LEARNING_RATE, 10);
      while(!train.step())
        kann::displayInfo("Training", train.info());

      std::cout << '\n';
    }

    // Testing
    {
      auto task = kann::test(model, testing_inputs, testing_outputs);
      while(!task.step())
        kann::displayInfo("Testing", task.info());

      std::cout << "correctness:" << task.get() << std::endl;
    }
  }
  else if(subcommand == "autoencoder")
  {
    if(argc != 3)
      return -1;

    auto encoder_layer = kann::loadLayer("examples/backpropagation/autoencoder/" + std::string(argv[2]) + "-encoder.yaml");
    auto decoder_layer = kann::loadLayer("examples/backpropagation/autoencoder/" + std::string(argv[2]) + "-decoder.yaml");

    auto auto_encoder_layer = std::make_shared<kann::SequentialLayer>();
    auto_encoder_layer->addLayer(encoder_layer, kann::Tag::ENCODER);
    auto_encoder_layer->addLayer(decoder_layer, kann::Tag::DECODER);
    auto_encoder_layer->randomize(engine);

    auto encoder_model = std::make_shared<kann::Model>(encoder_layer);
    auto decoder_model = std::make_shared<kann::Model>(decoder_layer);
    auto auto_encoder_model = std::make_shared<kann::Model>(auto_encoder_layer);

    decoder_model->compile(BATCH_SIZE, OPTIMIZER, {});
    encoder_model->compile(BATCH_SIZE, OPTIMIZER, {});
    auto_encoder_model->compile(BATCH_SIZE, OPTIMIZER, {kann::Tag::ALL});

    const std::string reconstruction_output_path = "output/autoencoder-reconstruction";
    const std::string output_path                = "output/autoencoder";

    std::filesystem::create_directories(reconstruction_output_path);
    std::filesystem::create_directories(output_path);

    // Training
    {
      auto task = kann::train(auto_encoder_model, training_inputs, training_inputs, LEARNING_RATE, BATCH_SIZE);
      while(!task.step())
        kann::displayInfo("Training", task.info());

      std::cout << '\n';
    }


    // Reconstruction
    {
      auto task = kann::run(auto_encoder_model, training_inputs);
      while(!task.step())
      {
        kann::Info info = task.info();
        kann::toImage(*info.output, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH)
          .saveToFile(reconstruction_output_path + "/result" + std::to_string(info.i) + ".bmp");
      }
    }

    // Generation
    {
      auto random_data = kann::load(kann::RandomDataSet(FEATURES_COUNT, 500), kann::RandomDataSet::COLUMN_DATA);
      auto task = kann::run(decoder_model, random_data);
      while(!task.step())
      {
        kann::Info info = task.info();
        kann::toImage(*info.output, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH)
          .saveToFile(output_path + "/result" + std::to_string(info.i) + ".bmp");
      }

    }
  }
  else if(subcommand == "gan")
  {
    auto generator_layer     = kann::loadLayer("examples/backpropagation/gan/generator.yaml");
    auto discriminator_layer = kann::loadLayer("examples/backpropagation/gan/discriminator.yaml");

    auto gan_layer = std::make_shared<kann::SequentialLayer>();
    gan_layer->addLayer(generator_layer    , kann::Tag::GAN_GENERATOR);
    gan_layer->addLayer(discriminator_layer, kann::Tag::GAN_DISCRIMINATOR);
    gan_layer->randomize(engine);

    auto generator_model = std::make_shared<kann::Model>(generator_layer);
    auto discriminator_model = std::make_shared<kann::Model>(discriminator_layer);
    auto gan_model = std::make_shared<kann::Model>(gan_layer);

    generator_model    ->compile(BATCH_SIZE, OPTIMIZER, {kann::Tag::ALL});
    discriminator_model->compile(BATCH_SIZE, OPTIMIZER, {});
    gan_model          ->compile(BATCH_SIZE, OPTIMIZER, {kann::Tag::GAN_GENERATOR, kann::Tag::GAN_DISCRIMINATOR});

    std::string output_directory("output/gan");
    std::filesystem::create_directories(output_directory + "/training");
    std::filesystem::create_directories(output_directory + "/output");

    // Training
    {
      auto latent_inputs = kann::load(kann::RandomDataSet(FEATURES_COUNT, training_inputs.size()), kann::RandomDataSet::COLUMN_DATA);
      auto task = kann::trainGAN(gan_model, generator_model, discriminator_model, training_inputs, latent_inputs, LEARNING_RATE, BATCH_SIZE);
      while(!task.step())
      {
        kann::GANInfo info = task.info();
        kann::toImage(*info.generatorOutput, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH)
          .saveToFile(output_directory + "/training" + std::to_string(info.i) + ".png");
        kann::displayInfo("Training", info);
      }
      std::cout << '\n';
    }

    // Generation
    {
      auto latent_inputs = kann::load(kann::RandomDataSet(FEATURES_COUNT, training_inputs.size()), kann::RandomDataSet::COLUMN_DATA);

      auto task = kann::run(generator_model, latent_inputs);
      while(!task.step())
      {
        kann::Info info = task.info();
        kann::toImage(*info.output, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH)
          .saveToFile(output_directory + "/output" + std::to_string(info.i) + ".png");
        kann::displayInfo("Training", info);
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
