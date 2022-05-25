#pragma once

#include <vector>
#include <memory>

namespace kann
{
  struct Tensor;

  static constexpr size_t MNIST_DATASET_IMAGE_WIDTH = 28;
  std::vector<std::shared_ptr<const Tensor>> load_mnist_dataset_images(const char* file_name);
  std::vector<std::shared_ptr<const Tensor>> load_mnist_dataset_labels(const char* file_name);
}
