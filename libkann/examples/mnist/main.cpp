#include <libkann/Initialize.hpp>

#include <libtensor/Tensor.hpp>
#include <libtensor/MaxCoeff.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>

#include <libkann/datasets/MNIST.hpp>
#include <libkann/ProgressBar.hpp>

#include <libtensor/memops/Stack.hpp>
#include <libtensor/memops/Split.hpp>
#include <libtensor/memops/Index.hpp>

#include <random>

#include <fmt/core.h>

#include "../common/common.hpp"

static void testing(std::string_view label, kann::Layer& layer, tensor::Tensor<float> images, tensor::Tensor<float> labels, size_t count)
{
  auto preds        = layer.forward(std::move(images));

  auto preds_single  = tensor::split_outer(std::move(preds));
  auto labels_single = tensor::split_outer(std::move(labels));
  auto correct_count = ranges::count_if(ranges::views::zip(preds_single, labels_single), [](const auto& p) {
    return tensor::max_coeff(p.first) == tensor::max_coeff(p.second);
  });
  fmt::print("  {} accuracy:{}/{}\n", label, correct_count, count);
}

static void training(kann::Layer& layer, kann::LossFunction& loss_function, tensor::Tensor<float> images, tensor::Tensor<float> labels, kann::Optimizer& optimizer, size_t batch_size, size_t count, auto& prng)
{
  std::vector<size_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), prng);

  kann::ProgressBar progress_bar("Training", count / batch_size * batch_size);
  for(size_t i=0; i+batch_size<=count; i+=batch_size)
  {
    auto sub_indices = std::vector(&indices[i], &indices[i+batch_size]);
    auto images_batch = tensor::index_outer(images, sub_indices);
    auto labels_batch = tensor::index_outer(labels, sub_indices);

    loss_function.shape = layer.get_output_shape();
    loss_function.expected_outputs = std::move(labels_batch);

    auto preds_batch = layer.forward(std::move(images_batch));
    auto losses_batch = loss_function.forward(std::move(preds_batch));
    auto ones_batch   = tensor::create_constant(tensor::Shape::make(batch_size), 1.0f);
    auto grads_batch = loss_function.backward(std::move(ones_batch));
    layer.backward(std::move(grads_batch));

    for(auto [name, parameter] : layer.parameters_map())
      optimizer.optimize(*parameter);

    optimizer.step();

    for(size_t i=0; i<batch_size; ++i)
      progress_bar.update(fmt::format("  loss={}", (*losses_batch.buffer)[i]));
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

  const auto layer = kann::LayerDef::load(file_name)->create();
  layer->initialize(prng);

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
  testing("Initial testing", *layer, mnist_testing_images, mnist_testing_labels, 10000);
  for(size_t i=0; i<epoch; ++i)
  {
    fmt::print("=> Epoch {} begin\n", i);
    {
      // Testing training
      training(*layer, *loss_function, mnist_training_images, mnist_training_labels, *optimizer, batch_size, 60000, prng);
      testing("Training dataset", *layer, mnist_training_images, mnist_training_labels, 60000);
      testing("Testing dataset",  *layer, mnist_testing_images,  mnist_testing_labels,  10000);
    }
    fmt::print("<= Epoch {} end\n", i);
  }
}
