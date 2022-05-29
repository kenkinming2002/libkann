#include "Renderer.hpp"
#include "Arguments.hpp"
#include "Progress.hpp"

#include <libkann/Random.hpp>

#include <libkann/Algorithm.hpp>
#include <libkann/Executor.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>

#include <libkann/layer_defs/Sequential.hpp>

#include <libkann/optimizers/AdamOptimizer.hpp>
#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libkann/datasets/MNIST.hpp>
#include <libkann/datasets/Random.hpp>
#include <libkann/datasets/write.hpp>

#include <range/v3/all.hpp>
#include <fmt/core.h>

#include <iostream>
#include <memory>
#include <random>
#include <filesystem>

static constexpr double LEARNING_RATE  = 0.05;
static constexpr size_t BATCH_SIZE     = 10;
static constexpr size_t FEATURES_COUNT = 64;

/* Note:
 *
 * Auto encoder and Adam Optimizer does not play nicely together - at
 * least when used in conjuction with sum of square difference cost function.
 * For better performance, use Simple Optimizer with auto encoder. */
//static const auto OPTIMIZER      = std::make_shared<kann::AdamOptimizer>(0.001, 0.9, 0.999, 1e-10);
static const auto OPTIMIZER = std::make_shared<kann::SimpleOptimizer>(0.05);
static const auto EXECUTOR  = kann::Executor::create(kann::Executor::Type::DEFAULT);

void usage()
{
  std::cerr << "Usage: backpropagation write/feedForward/autoencoder [target] [--gui]" << std::endl;
}

static inline bool is_correct(const kann::Tensor& value, const kann::Tensor& expected)
{
  size_t index1, index2;
  value.asVector().maxCoeff(&index1);
  expected.asVector().maxCoeff(&index2);
  return index1 == index2;
}

static inline double cost(const kann::Tensor& value, const kann::Tensor& expected)
{
  return (expected.asVector() - value.asVector()).squaredNorm();
}

int main(int argc, char** argv)
{
  // 1: Commandline
  std::optional<std::string_view> subcommand;
  std::optional<std::string_view> target;
  bool gui = false;

  bool result = Arguments(argc, argv).parse([&](Arguments::Result result) -> bool
  {
    switch(result.type)
    {
    case Arguments::Type::SHORT_OPTION:
      if(result.c == 'g')
      {
        gui = true;
        return true;
      }
      else
        return false;
    case Arguments::Type::LONG_OPTION:
      if(result.str == "gui")
      {
        gui = true;
        return true;
      }
      else
        return false;
    case Arguments::Type::OTHER:
      if(!subcommand)
      {
        subcommand = result.str;
        return true;
      }
      else if(!target)
      {
        target = result.str;
        return true;
      }
      else
      {
        std::cout << "Hey3\n";
        return false;
      }
    default:
      return false;
    }
  });

  if(!result)
  {
    usage();
    return -1;
  }

  if(!subcommand)
  {
    usage();
    return -1;
  }

  // 2: Launch GUI Thread
  std::optional<Renderer> renderer;
  if(gui)
    renderer.emplace();

  // 3: Computation
  std::default_random_engine engine(kann::random<std::default_random_engine::result_type>());

  auto training_images = kann::load_mnist_dataset_images("datasets/mnist/train-images-idx3-ubyte");
  auto training_labels = kann::load_mnist_dataset_labels("datasets/mnist/train-labels-idx1-ubyte");

  auto testing_images = kann::load_mnist_dataset_images("datasets/mnist/t10k-images-idx3-ubyte");
  auto testing_labels = kann::load_mnist_dataset_labels("datasets/mnist/t10k-labels-idx1-ubyte");

  if(*subcommand == "write")
  {
    for(const auto& [type, data] : { std::pair{"testing", testing_images}, std::pair{"training", training_images} })
    {
      auto dir_path  = fmt::format("output/{}", type);
      std::filesystem::create_directories(dir_path);

      for(const auto& [i, datum] : ranges::views::enumerate(data))
      {
        auto file_path = fmt::format("output/{}/{}.bmp", type, i);
        kann::toImage(*datum, kann::MNIST_DATASET_IMAGE_WIDTH, kann::MNIST_DATASET_IMAGE_WIDTH).saveToFile(file_path);
      }
    }
  }
  else if(*subcommand == "feedforward")
  {
    if(!target)
      return -1;

    auto layer = kann::LayerDef::load(fmt::format("examples/backpropagation/feedforward/{}.yaml", *target))->create(engine);

    auto test = [&]()
    {
      ProgressBar progress_bar("Testing", 10000);

      size_t i = 0;
      size_t correct_count = 0;
      for(auto task = kann::predict(*layer, testing_images, BATCH_SIZE, *EXECUTOR); !task.is_done(); task.step())
      {
        kann::PredictInfo info = task.info();
        const auto& expected_output = *testing_labels[i++];
        correct_count += (int)is_correct(info.output, expected_output);
        progress_bar.update(fmt::format("cost={}", cost(info.output, expected_output)));
      }

      fmt::print("correctness:{}\n", (double)correct_count/10000);
    };

    auto train = [&]()
    {
      ProgressBar progress_bar("Testing", 60000);
      for(auto task = kann::optimize(*layer, kann::Tag::DEFAULT, OPTIMIZER, training_images, training_labels, BATCH_SIZE, *EXECUTOR); !task.is_done(); task.step())
      {
        kann::OptimizeInfo info = task.info();
        progress_bar.update(fmt::format("cost={}", info.cost));
      }
    };

    test();
    train();
    test();
  }
  else if(*subcommand == "autoencoder")
  {
    if(!target)
      return -1;

    auto layer_def = std::make_shared<kann::SequentialLayerDef>();
    layer_def->sub_layer_defs = {
      kann::LayerDef::load(fmt::format("examples/backpropagation/autoencoder/{}-encoder.yaml", *target)),
      kann::LayerDef::load(fmt::format("examples/backpropagation/autoencoder/{}-decoder.yaml", *target))
    };
    auto layer         = layer_def->create(engine);
    auto decoder_layer = layer->sub_layers[1];

    // Testing
    {
      ProgressBar progress_bar("Testing", 60000);

      size_t i = 0;
      for(auto task = kann::optimize(*layer, kann::Tag::DEFAULT, OPTIMIZER, training_images, training_images, BATCH_SIZE, *EXECUTOR); !task.is_done(); task.step())
      {
        kann::OptimizeInfo info = task.info();
        progress_bar.update(fmt::format("cost={}", info.cost));
        if(renderer)
        {
          Renderer::Content content;
          content.title = fmt::format("{}/60000", ++i);
          content.images = {
            kann::toImage(info.input,  kann::MNIST_DATASET_IMAGE_WIDTH, kann::MNIST_DATASET_IMAGE_WIDTH),
            kann::toImage(info.output, kann::MNIST_DATASET_IMAGE_WIDTH, kann::MNIST_DATASET_IMAGE_WIDTH)
          };
          renderer->submit(content);
        }
      }
    }

    // Generation
    {
      auto latent_data = kann::create_random_data(FEATURES_COUNT, 500);
      std::filesystem::create_directories("output/autoencoder");

      ProgressBar progress_bar("Generation", 500);

      size_t i = 0;
      for(auto task = kann::predict(*decoder_layer, latent_data, BATCH_SIZE, *EXECUTOR); !task.is_done(); task.step())
      {
        kann::PredictInfo info = task.info();
        progress_bar.update("");
        kann::toImage(info.output, kann::MNIST_DATASET_IMAGE_WIDTH, kann::MNIST_DATASET_IMAGE_WIDTH)
          .saveToFile(fmt::format("output/autoencoder/{}.bmp", ++i));
      }
    }
  }
  else
  {
    std::cerr << "Error: Invalid Command " << *subcommand << std::endl;
    usage();
    return -1;
  }
}
