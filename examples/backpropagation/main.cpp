#include "Renderer.hpp"
#include "Arguments.hpp"
#include "Progress.hpp"

#include <libkann/Initialize.hpp>
#include <libkann/Random.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/Graph.hpp>
#include <libkann/Executor.hpp>

#include <libkann/optimizers/SimpleOptimizer.hpp>
#include <libkann/layer_defs/Sequential.hpp>

#include <libkann/datasets/MNIST.hpp>
#include <libkann/datasets/Random.hpp>
#include <libkann/datasets/write.hpp>

#include <libkann/Algorithm.hpp>

#include <range/v3/all.hpp>


static constexpr double LEARNING_RATE  = 0.05;
static constexpr size_t BATCH_SIZE     = 10;
static constexpr size_t FEATURES_COUNT = 64;

static bool correct(const kann::tensor_t& value1, const kann::tensor_t& value2)
{
  size_t index1, index2;
  value1->asArray().maxCoeff(&index1);
  value2->asArray().maxCoeff(&index2);
  return index1 == index2;
}

void run(const std::string& filename)
{
  static std::default_random_engine prng(kann::random<std::default_random_engine::result_type>());
  static auto executor  = kann::Executor::create(kann::Executor::Type::DEFAULT);
  static auto optimizer = std::make_shared<kann::SimpleOptimizer>(LEARNING_RATE);

  static auto testing_images = kann::load_mnist_dataset_images("datasets/mnist/t10k-images-idx3-ubyte");
  static auto testing_labels = kann::load_mnist_dataset_labels("datasets/mnist/t10k-labels-idx1-ubyte");

  static auto training_images = kann::load_mnist_dataset_images("datasets/mnist/train-images-idx3-ubyte");
  static auto training_labels = kann::load_mnist_dataset_labels("datasets/mnist/train-labels-idx1-ubyte");

  auto layer = kann::LayerDef::load(filename)->create(prng);

  // Testing
  {
    auto predictions = kann::predict(*layer, *executor, testing_images);
    size_t correct_count = ranges::count_if(ranges::views::zip(testing_labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
    fmt::print("Accuracy:{}/10000\n", correct_count);
  }

  // Training
  kann::optimize(*layer, kann::Tag::ALL, *optimizer, *executor, training_images, training_labels);

  // Testing on training set
  {
    auto predictions = kann::predict(*layer, *executor, training_images);
    size_t correct_count = ranges::count_if(ranges::views::zip(training_labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
    fmt::print("Accuracy:{}/60000\n", correct_count);
  }

  // Testing on testing set
  {
    auto predictions = kann::predict(*layer, *executor, testing_images);
    size_t correct_count = ranges::count_if(ranges::views::zip(testing_labels, predictions), [](const auto& p){ return correct(p.first, p.second); });
    fmt::print("Accuracy:{}/10000\n", correct_count);
  }
}

int main(int argc, char** argv)
{
  kann::initialize();
  run("examples/backpropagation/feedforward/normal.yaml");
  run("examples/backpropagation/feedforward/convolution.yaml");
}
