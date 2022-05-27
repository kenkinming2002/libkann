#pragma once

#include <libkann/Types.hpp>

#include <vector>

namespace kann
{
  static constexpr size_t MNIST_DATASET_IMAGE_WIDTH = 28;
  std::vector<tensor_t> load_mnist_dataset_images(const char* file_name);
  std::vector<tensor_t> load_mnist_dataset_labels(const char* file_name);
}
