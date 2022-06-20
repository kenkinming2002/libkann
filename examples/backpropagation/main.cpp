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

#include <libkann/datasets/MNIST.hpp>
#include <libkann/datasets/Random.hpp>
#include <libkann/datasets/write.hpp>

#include <libkann/Batch.hpp>

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

  const size_t batch_size = std::stoull(batch_size_str);
  const size_t epoch      = std::stoull(epoch_str);

  // 3: Running
  const std::vector<kann::Tensor> mnist_testing_images = kann::load_mnist_dataset_images("datasets/mnist/t10k-images-idx3-ubyte");
  const std::vector<kann::Tensor> mnist_testing_labels = kann::load_mnist_dataset_labels("datasets/mnist/t10k-labels-idx1-ubyte");

  const std::vector<kann::Tensor> mnist_training_images = kann::load_mnist_dataset_images("datasets/mnist/train-images-idx3-ubyte");
  const std::vector<kann::Tensor> mnist_training_labels = kann::load_mnist_dataset_labels("datasets/mnist/train-labels-idx1-ubyte");

  // Initial testing
  {
    const std::vector<kann::Tensor>& images = mnist_testing_images;
    const std::vector<kann::Tensor>& labels = mnist_testing_labels;

    const std::vector<kann::Tensor>& image_batches = kann::batch(images, batch_size);
    const std::vector<kann::Tensor>& prediction_batches = image_batches
      | ranges::views::transform([&layer](const kann::Tensor& image_batch) { return layer->forward(image_batch); })
      | ranges::to_vector;

    const std::vector<kann::Tensor>& predictions = kann::unbatch(prediction_batches, batch_size);
    size_t correct_count = ranges::count_if(ranges::views::zip(labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
    fmt::print("Initial testing accuracy:{}/10000\n", correct_count);
  }

  for(size_t i=0; i<epoch; ++i)
  {
    fmt::print("=> Epoch {} begin\n", i);
    {
      // Training
      {
        std::vector<kann::Tensor> images = mnist_training_images;
        std::vector<kann::Tensor> labels = mnist_training_labels;

        // Shuffle
        {
          std::uniform_int_distribution<size_t> dist(0, 60000-1);
          for(size_t i=0; i<60000; ++i)
          {
            size_t index1 = dist(prng), index2 = dist(prng);
            if(index1 == index2)
              continue;

            std::swap(images[index1], images[index2]);
            std::swap(labels[index1], labels[index2]);
          }
        }

        const std::vector<kann::Tensor>& image_batches = kann::batch(images, batch_size);
        const std::vector<kann::Tensor>& label_batches = kann::batch(labels, batch_size);
        for(const auto& [image_batch, label_batch] : ranges::views::zip(image_batches, label_batches))
        {
          kann::Tensor prediction_batch = layer->forward(image_batch);
          kann::Tensor gradient_batch = kann::math::cwise(label_batch, prediction_batch, [](double label_value, double prediction_value) {
            return prediction_value - label_value;
          });
          layer->backward(gradient_batch);
          layer->storage->foreach_parameters([&optimizer](kann::Variable& variable) { optimizer->optimize(variable); });
          optimizer->step();
        }
      }

      // Testing training
      {
        const std::vector<kann::Tensor>& images = mnist_training_images;
        const std::vector<kann::Tensor>& labels = mnist_training_labels;

        const std::vector<kann::Tensor>& image_batches = kann::batch(images, batch_size);
        const std::vector<kann::Tensor>& prediction_batches = image_batches
          | ranges::views::transform([&layer](const kann::Tensor& image_batch) { return layer->forward(image_batch); })
          | ranges::to_vector;

        const std::vector<kann::Tensor>& predictions = kann::unbatch(prediction_batches, batch_size);
        size_t correct_count = ranges::count_if(ranges::views::zip(labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
        fmt::print("Training dataset accuracy:{}/10000\n", correct_count);
      }

      // Testing testing
      {
        const std::vector<kann::Tensor>& images = mnist_testing_images;
        const std::vector<kann::Tensor>& labels = mnist_testing_labels;

        const std::vector<kann::Tensor>& image_batches = kann::batch(images, batch_size);
        const std::vector<kann::Tensor>& prediction_batches = image_batches
          | ranges::views::transform([&layer](const kann::Tensor& image_batch) { return layer->forward(image_batch); })
          | ranges::to_vector;

        const std::vector<kann::Tensor>& predictions = kann::unbatch(prediction_batches, batch_size);
        size_t correct_count = ranges::count_if(ranges::views::zip(labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
        fmt::print("Testing dataset accuracy:{}/10000\n", correct_count);
      }
    }
    fmt::print("<= Epoch {} end\n", i);
  }
}
