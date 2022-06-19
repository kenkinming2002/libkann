#include <libkann/Initialize.hpp>
#include <libkann/Random.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Utils.hpp>
#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/Graph.hpp>

#include <libkann/executors/DefaultExecutor.hpp>

#include <libkann/optimizers/AdamOptimizer.hpp>
#include <libkann/optimizers/SimpleOptimizer.hpp>

#include <libkann/datasets/MNIST.hpp>
#include <libkann/datasets/Random.hpp>
#include <libkann/datasets/write.hpp>

#include <libkann/Algorithm.hpp>

#include <fmt/core.h>

static bool correct(const kann::Tensor& value1, const kann::Tensor& value2)
{
  return kann::utils::max_coeff(value1.as_ref()) == kann::utils::max_coeff(value2.as_ref());
}

int main(int argc, char** argv)
{
  // 1: Commandline arguments parsing
  if(argc != 6)
  {
    fmt::print("Usage: {} [FILE_NAME] [OPTIMIZER_NAME] [OPTIMIZER_PARAMETERS] [BATCH_SIZE] [EPOCH]", argv[0]);
    return EXIT_FAILURE;
  }
  std::string file_name            = argv[1];
  std::string optimizer_name       = argv[2];
  std::string optimizer_parameters = argv[3];
  std::string batch_size_str       = argv[4];
  std::string epoch_str            = argv[5];

  fmt::print("DEBUG: file_name            = {}\n", file_name);
  fmt::print("DEBUG: optimizer_name       = {}\n", optimizer_name);
  fmt::print("DEBUG: optimizer_parameters = {}\n", optimizer_parameters);
  fmt::print("DEBUG: batch_size           = {}\n", batch_size_str);
  fmt::print("DEBUG: epoch                = {}\n", epoch_str);

  // 2: Preparation
  kann::initialize();

  std::default_random_engine prng(kann::random<std::default_random_engine::result_type>());

  const kann::layer_t layer = kann::LayerDef::load(file_name)->create(prng);
  const kann::optimizer_t optimizer = [&optimizer_name, &optimizer_parameters]() -> kann::optimizer_t
  {
    if(optimizer_name == "simple")
    {
      const float learning_rate = std::stof(optimizer_parameters);
      return std::make_shared<kann::SimpleOptimizer>(learning_rate);
    }
    else if(optimizer_name == "adam")
    {
      std::stringstream ss(optimizer_parameters);
      auto next = [&]() -> float
      {
        std::string str;
        if(!std::getline(ss, str, ','))
          throw std::runtime_error(fmt::format("Adam optimizer:Invalid parameters:{}", optimizer_parameters));

        return std::stof(str);
      };

      const float alpha = next(), beta1 = next(), beta2 = next(), epsilon = next();
      return std::make_shared<kann::AdamOptimizer>(alpha, beta1, beta2, epsilon);
    }
    else
      throw std::runtime_error(fmt::format("Unknown optimizer name:{}", optimizer_name));
  }();

  const size_t batch_size = std::stoull(batch_size_str);
  const size_t epoch      = std::stoull(epoch_str);

  // 3: Running
  const std::vector<kann::Tensor> testing_images = kann::load_mnist_dataset_images("datasets/mnist/t10k-images-idx3-ubyte");
  const std::vector<kann::Tensor> testing_labels = kann::load_mnist_dataset_labels("datasets/mnist/t10k-labels-idx1-ubyte");

  const std::vector<kann::Tensor> training_images = kann::load_mnist_dataset_images("datasets/mnist/train-images-idx3-ubyte");
  const std::vector<kann::Tensor> training_labels = kann::load_mnist_dataset_labels("datasets/mnist/train-labels-idx1-ubyte");

  const std::shared_ptr<kann::Executor> executor = std::make_shared<kann::DefaultExecutor>();

  // Testing
  {
    auto predictions = kann::predict(*layer, *executor, testing_images);
    size_t correct_count = ranges::count_if(ranges::views::zip(testing_labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
    fmt::print("Initial testing accuracy:{}/10000\n", correct_count);
  }

  // Training
  for(size_t i=0; i<epoch; ++i)
  {
    fmt::print("=> Epoch {} begin\n", i);
    {
      kann::optimize(*layer, kann::Tag::ALL, *optimizer, *executor, batch_size, training_images, training_labels);

      // Testing on training set
      {
        auto predictions = kann::predict(*layer, *executor, training_images);
        size_t correct_count = ranges::count_if(ranges::views::zip(training_labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
        fmt::print("  Training set accuracy:{}/60000\n", correct_count);
      }

      // Testing on testing set
      {
        auto predictions = kann::predict(*layer, *executor, testing_images);
        size_t correct_count = ranges::count_if(ranges::views::zip(testing_labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
        fmt::print("  Testing set accurracy:{}/10000\n", correct_count);
      }
    }
    fmt::print("<= Epoch {} end\n", i);
  }
}
