#include <libkann/Initialize.hpp>

#include <libtensor/Tensor.hpp>
#include <libtensor/MaxCoeff.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerStorage.hpp>
#include <libkann/LayerDef.hpp>

#include <libkann/datasets/MNIST.hpp>

#include <libkann/Batch.hpp>
#include <libkann/ProgressBar.hpp>

#include <random>

#include <fmt/core.h>

#include "../common/common.hpp"

static bool correct(const tensor::Tensor<float>& value1, const tensor::Tensor<float>& value2)
{
  return tensor::max_coeff(value1) == tensor::max_coeff(value2);
}

static void testing(std::string_view label, kann::Layer& layer,
    std::vector<tensor::Tensor<float>> images,
    std::vector<tensor::Tensor<float>> labels,
    size_t batch_size, size_t count)
{
  kann::ProgressBar progress_bar("  testing", count / batch_size);

  auto image_batches = kann::batch(images, batch_size);
  auto label_batches = kann::batch(labels, batch_size);

  size_t correct_count = 0;
  for(const auto& [image_batch, label_batch] : ranges::views::zip(image_batches, label_batches))
  {
    auto pred_batch = layer.forward(image_batch);
    auto labels = kann::unbatch({label_batch}, batch_size);
    auto preds  = kann::unbatch({pred_batch},  batch_size);
    correct_count += ranges::count_if(ranges::views::zip(labels, preds), [](const auto& p){ return correct(p.first, p.second); });
    progress_bar.update("", 1);
  }
  fmt::print("  {} accuracy:{}/{}\n", label, correct_count, count);
}

static void training(kann::Layer& layer, kann::LossFunction& loss_function,
    std::vector<tensor::Tensor<float>> images,
    std::vector<tensor::Tensor<float>> labels,
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

  auto image_batches = kann::batch(images, batch_size);
  auto label_batches = kann::batch(labels, batch_size);
  for(auto&& [image_batch, label_batch] : ranges::views::zip(image_batches, label_batches))
  {
    loss_function.shape = layer.def->get_output_shape();
    loss_function.expected_outputs = std::move(label_batch);

    auto ones_batch = tensor::create_constant(tensor::Shape::make(batch_size), 1.0f);
    auto pred_batch = layer.forward(std::move(image_batch));
    auto loss_batch = loss_function.forward(std::move(pred_batch));
    auto grad_batch = loss_function.backward(std::move(ones_batch));
    layer.backward(std::move(grad_batch));

    for(size_t i=0; i<batch_size; ++i)
      progress_bar.update(fmt::format("  loss={}", loss_batch.buffer->data()[i]));

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
    fmt::print("Usage: {} [FILE_NAME] [OPTIMIZER_NAME] [OPTIMIZER_ARG] [LOSS_FUNCTION] [LOSS_FUNCTION_ARG] [BATCH_SIZE] [EPOCH]", argv[0]);
    return EXIT_FAILURE;
  }

  const char* file_name          = argv[1];
  const char* optimizer_name     = argv[2];
  const char* optimizer_arg      = argv[3];
  const char* loss_function_name = argv[4];
  const char* loss_function_arg  = argv[5];
  const char* batch_size_str     = argv[6];
  const char* epoch_str          = argv[7];

  fmt::print("DEBUG: file_name          = {}\n", file_name);
  fmt::print("DEBUG: optimizer_name     = {}\n", optimizer_name);
  fmt::print("DEBUG: optimizer_arg      = {}\n", optimizer_arg);
  fmt::print("DEBUG: loss_function_name = {}\n", loss_function_name);
  fmt::print("DEBUG: loss_function_arg  = {}\n", loss_function_arg);
  fmt::print("DEBUG: batch_size         = {}\n", batch_size_str);
  fmt::print("DEBUG: epoch              = {}\n", epoch_str);

  // 2: Preparation
  kann::initialize();

  std::random_device rd;
  std::default_random_engine prng(rd());

  const auto def           = kann::LayerDef::load(file_name);
  const auto storage       = def->create(prng);
  const auto layer         = kann::Layer::create_from(def, storage);
  const auto optimizer     = create_optimizer(optimizer_name, optimizer_arg);
  const auto loss_function = create_loss_function(loss_function_name, loss_function_arg);
  const size_t batch_size = std::stoull(batch_size_str);
  const size_t epoch      = std::stoull(epoch_str);

  // 3: Running
  auto mnist_testing_images = kann::load_mnist_dataset_images("libkann/datasets/mnist/t10k-images-idx3-ubyte");
  auto mnist_testing_labels = kann::load_mnist_dataset_labels("libkann/datasets/mnist/t10k-labels-idx1-ubyte");

  auto mnist_training_images = kann::load_mnist_dataset_images("libkann/datasets/mnist/train-images-idx3-ubyte");
  auto mnist_training_labels = kann::load_mnist_dataset_labels("libkann/datasets/mnist/train-labels-idx1-ubyte");

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
