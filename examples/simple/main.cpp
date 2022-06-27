#include <libkann/Initialize.hpp>
#include <libkann/Random.hpp>

#include <libkann/Tensor.hpp>
#include <libkann/Math.hpp>
#include <libkann/Utils.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerStorage.hpp>
#include <libkann/LayerDef.hpp>

#include <libkann/optimizers/SimpleOptimizer.hpp>
#include <libkann/optimizers/AdamOptimizer.hpp>

#include <libkann/loss_functions/Lp.hpp>
#include <libkann/loss_functions/CrossEntropy.hpp>

#include <libkann/datasets/MNIST.hpp>
#include <libkann/datasets/Random.hpp>
#include <libkann/datasets/write.hpp>

#include <libkann/Batch.hpp>
#include <libkann/ProgressBar.hpp>

#include <fmt/core.h>

static bool correct(const kann::Tensor<float>& value1, const kann::Tensor<float>& value2)
{
  return kann::utils::max_coeff(value1) == kann::utils::max_coeff(value2);
}

static void testing(std::string_view label, kann::Layer& layer,
    std::vector<kann::Tensor<float>> images,
    std::vector<kann::Tensor<float>> labels,
    size_t batch_size, size_t count)
{
  assert(images.size() == count);
  assert(labels.size() == count);

  // Testing
  std::vector<kann::Tensor<float>> image_batches = kann::batch(images, batch_size);
  std::vector<kann::Tensor<float>> prediction_batches;

  kann::ProgressBar progress_bar("  testing", count);
  for(kann::Tensor<float>& image_batch : image_batches)
  {
    prediction_batches.push_back(layer.forward(std::move(image_batch)));
    progress_bar.update("", batch_size);
  }

  std::vector<kann::Tensor<float>> predictions = kann::unbatch(prediction_batches, batch_size);

  // Report
  size_t correct_count = ranges::count_if(ranges::views::zip(labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
  fmt::print("  {} accuracy:{}/{}\n", label, correct_count, count);
}

static void training(kann::Layer& layer, kann::LossFunction& loss_function,
    std::vector<kann::Tensor<float>> images,
    std::vector<kann::Tensor<float>> labels,
    kann::Optimizer& optimizer,
    size_t batch_size, size_t count, auto& prng)
{
  assert(images.size() == count);
  assert(labels.size() == count);

  std::uniform_int_distribution<size_t> dist(0, count-1);
  for(size_t i=0; i<count; ++i)
  {
    size_t index1 = dist(prng), index2 = dist(prng);
    if(index1 == index2)
      continue;

    std::swap(images[index1], images[index2]);
    std::swap(labels[index1], labels[index2]);
  }

  kann::ProgressBar progress_bar("  training", count);

  std::vector<kann::Tensor<float>> image_batches = kann::batch(images, batch_size);
  std::vector<kann::Tensor<float>> label_batches = kann::batch(labels, batch_size);
  for(auto&& [image_batch, label_batch] : ranges::views::zip(image_batches, label_batches))
  {
    kann::Tensor<float> prediction_batch = layer.forward(std::move(image_batch));

    loss_function.expected_outputs = std::move(label_batch);
    kann::Tensor<float> loss_batch = loss_function.forward(std::move(prediction_batch));

    kann::Tensor<float> ones_batch = kann::Tensor<float>::create(kann::Shape(batch_size));
    ones_batch.fill(1.0);

    kann::Tensor<float> gradient_batch = loss_function.backward(std::move(ones_batch));
    layer.backward(std::move(gradient_batch));

    for(size_t i=0; i<batch_size; ++i)
      progress_bar.update(fmt::format("  loss={}", loss_batch(i)));

    for(kann::Variable* parameter : layer.storage->get_parameters())
      optimizer.optimize(*parameter);

    optimizer.step();
  }
}

int main(int argc, char** argv)
{
  // 1: Commandline arguments parsing
  if(argc != 8)
  {
    fmt::print("Usage: {} [FILE_NAME] [OPTIMIZER_NAME] [OPTIMIZER_PARAMETERS] [LOSS_FUNCTION] [LOSS_FUNCTION_PARAMETERS] [BATCH_SIZE] [EPOCH]", argv[0]);
    return EXIT_FAILURE;
  }
  std::string file_name                = argv[1];
  std::string optimizer_name           = argv[2];
  std::string optimizer_parameters     = argv[3];
  std::string loss_function_name       = argv[4];
  std::string loss_function_parameters = argv[5];
  std::string batch_size_str           = argv[6];
  std::string epoch_str                = argv[7];

  fmt::print("DEBUG: file_name                = {}\n", file_name);
  fmt::print("DEBUG: optimizer_name           = {}\n", optimizer_name);
  fmt::print("DEBUG: optimizer_parameters     = {}\n", optimizer_parameters);
  fmt::print("DEBUG: loss_function_name       = {}\n", loss_function_name);
  fmt::print("DEBUG: loss_function_parameters = {}\n", loss_function_parameters);
  fmt::print("DEBUG: batch_size               = {}\n", batch_size_str);
  fmt::print("DEBUG: epoch                    = {}\n", epoch_str);

  // 2: Preparation
  kann::initialize();

  std::default_random_engine prng(kann::random<std::default_random_engine::result_type>());

  const std::shared_ptr<const kann::LayerDef> def   = kann::LayerDef::load(file_name);
  const std::shared_ptr<kann::LayerStorage> storage = def->create(prng);
  const std::shared_ptr<kann::Layer> layer          = kann::Layer::create_from(def, storage);

  std::shared_ptr<kann::Optimizer> optimizer = [&optimizer_name, &optimizer_parameters]() -> std::shared_ptr<kann::Optimizer>
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

  std::shared_ptr<kann::LossFunction> loss_function = [&loss_function_name, &loss_function_parameters]() -> std::shared_ptr<kann::LossFunction>
  {
    if(loss_function_name == "lp")
    {
      const unsigned p = std::stoi(loss_function_parameters);
      return std::make_shared<kann::LpLossFunction>(p);
    }
    else if(loss_function_name == "cross_entropy")
    {
      if(loss_function_parameters != "none")
        throw std::runtime_error(fmt::format("Cross entropy loss function:Invalid parameters:{}:Expected 'none'", loss_function_parameters));

      return std::make_shared<kann::CrossEntropyLossFunction>();
    }
    else
      throw std::runtime_error(fmt::format("Unknown loss function name:{}", loss_function_name));
  }();

  const size_t batch_size = std::stoull(batch_size_str);
  const size_t epoch      = std::stoull(epoch_str);

  // 3: Running
  std::vector<kann::Tensor<float>> mnist_testing_images = kann::load_mnist_dataset_images("datasets/mnist/t10k-images-idx3-ubyte");
  std::vector<kann::Tensor<float>> mnist_testing_labels = kann::load_mnist_dataset_labels("datasets/mnist/t10k-labels-idx1-ubyte");

  std::vector<kann::Tensor<float>> mnist_training_images = kann::load_mnist_dataset_images("datasets/mnist/train-images-idx3-ubyte");
  std::vector<kann::Tensor<float>> mnist_training_labels = kann::load_mnist_dataset_labels("datasets/mnist/train-labels-idx1-ubyte");

  // Initial testing
  testing("Initial testing", *layer, mnist_testing_images, mnist_testing_labels, batch_size, 10000);
  for(size_t i=0; i<epoch; ++i)
  {
    fmt::print("=> Epoch {} begin\n", i);
    {
      // Testing training
      training(*layer, *loss_function, mnist_training_images, mnist_training_labels, *optimizer, batch_size, 60000, prng);
      testing("Training dataset", *layer, mnist_training_images, mnist_training_labels, batch_size, 60000);
      testing("Testing dataset", *layer, mnist_testing_images,  mnist_testing_labels,  batch_size, 10000);
    }
    fmt::print("<= Epoch {} end\n", i);
  }
}
