#include <libkann/Initialize.hpp>

#include <libtensor/Tensor.hpp>
#include <libtensor/MaxCoeff.hpp>

#include <libkann/SL.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/layers/Sequential.hpp>

#include <libkann/datasets/MNIST.hpp>
#include <libkann/ProgressBar.hpp>

#include <libtensor/memops/Stack.hpp>
#include <libtensor/memops/Split.hpp>
#include <libtensor/memops/Index.hpp>

#include <filesystem>
#include <fstream>
#include <random>

#include <fmt/core.h>

#include "../common/common.hpp"

static void reconstruct(std::string label, kann::Layer& layer, tensor::Tensor<float> images)
{
  auto preds        = layer.forward(std::move(images));
  auto preds_single  = tensor::split_outer(std::move(preds));

  size_t i = 0;

  std::filesystem::create_directories(fmt::format("output/{}", label));
  for(auto&& pred : preds_single)
  {
    // Write ppm file
    std::ofstream file(fmt::format("output/{}/{:05}.ppm", label, i++));
    file << "P3\n";
    file << "28 28\n";
    file << "255\n";
    for(size_t y=0; y<28; ++y)
    {
      for(size_t x=0; x<28; ++x)
      {
        const float value_f = (*pred.buffer)[y*28+x];
        const unsigned value_u = static_cast<unsigned>(std::max(value_f * 255.0f, 0.0f));

        if(x != 0) file << " ";
        file << value_u << " " << value_u << " " << value_u;
      }
      file << '\n';
    }
  }
}

static void generate(std::string label, kann::Layer& layer, size_t count, auto& prng)
{
  // Rely on the bottleneck layer having two node
  std::vector<tensor::Tensor<float>> images;
  images.reserve(count*count);

  for(size_t j=0; j<count; ++j)
    for(size_t i=0; i<count; ++i)
    {
      const float y = (float)j / (float)count;
      const float x = (float)i / (float)count;

      auto buffer = tensor::Buffer<float>(2);
      buffer[0] = y;
      buffer[1] = x;

      auto latent = tensor::Tensor<float>(tensor::Shape::make(1, 2), std::make_shared<tensor::Buffer<float>>(std::move(buffer)));
      auto image  = layer.forward(std::move(latent));
      images.push_back(std::move(image));
    }

  // Write ppm file
  std::filesystem::create_directories("output");
  std::ofstream file(fmt::format("output/{}.ppm", label));
  file << "P3\n";
  file << 28 * count << " " << 28 * count << "\n";
  file << "255\n";
  for(size_t j=0; j<count; ++j)
    for(size_t y=0; y<28; ++y)
    {
      for(size_t i=0; i<count; ++i)
        for(size_t x=0; x<28; ++x)
        {
          const float value_f = (*images[j*count+i].buffer)[y*28+x];
          const unsigned value_u = static_cast<unsigned>(std::max(value_f * 255.0f, 0.0f));

          if(i !=0 || x != 0) file << " ";
          file << value_u << " " << value_u << " " << value_u;
        }

      file << '\n';
    }
}

static void training(kann::Layer& layer, kann::LossFunction& loss_function, tensor::Tensor<float> images, kann::Optimizer& optimizer, size_t batch_size, size_t count, auto& prng)
{
  std::vector<size_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), prng);

  kann::ProgressBar progress_bar("Training", count / batch_size * batch_size);
  for(size_t i=0; i+batch_size<=count; i+=batch_size)
  {
    auto sub_indices = std::vector(&indices[i], &indices[i+batch_size]);
    auto images_batch = tensor::index_outer(images, sub_indices);

    loss_function.expected_outputs = images_batch;

    auto preds_batch = layer.forward(std::move(images_batch));
    auto losses_batch = loss_function.forward(std::move(preds_batch));
    auto ones_batch   = tensor::create_constant(tensor::Shape::make(batch_size), 1.0f);
    auto grads_batch = loss_function.backward(std::move(ones_batch));
    layer.backward(std::move(grads_batch));

    for(const auto&[name, parameter] : layer.parameters_map())
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

  const auto layer = kann::load_layer_def(file_name)->create();
  layer->initialize(prng);
  const auto encoder   = static_cast<kann::SequentialLayer*>(layer.get())->layers.at(0);
  const auto decoder   = static_cast<kann::SequentialLayer*>(layer.get())->layers.at(1);

  const auto optimizer     = create_optimizer(optimizer_name, optimizer_arg);
  const auto loss_function = create_loss_function(loss_function_name, loss_function_arg);
  const size_t batch_size = std::stoull(batch_size_str);
  const size_t epoch      = std::stoull(epoch_str);

  // 3: Running
  auto mnist_testing_images  = kann::load_mnist_dataset_images("libkann/datasets/mnist/t10k-images-idx3-ubyte") .reshape(tensor::Shape::make(10000, 1, 28, 28));
  auto mnist_training_images = kann::load_mnist_dataset_images("libkann/datasets/mnist/train-images-idx3-ubyte").reshape(tensor::Shape::make(60000, 1, 28, 28));

  // Initial testing
  for(size_t i=0; i<epoch; ++i)
  {
    fmt::print("=> Epoch {} begin\n", i);
    training(*layer, *loss_function, mnist_training_images, *optimizer, batch_size, 60000, prng);
    fmt::print("<= Epoch {} end\n", i);
  }
  reconstruct("training", *layer, mnist_training_images);
  reconstruct("testing",  *layer, mnist_testing_images);
  generate("generated", *decoder, 10, prng);
}
