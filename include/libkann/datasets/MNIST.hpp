#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  static constexpr size_t MNIST_DATASET_IMAGE_WIDTH = 28;
  std::vector<Tensor> load_mnist_dataset_images(const char* file_name);
  std::vector<Tensor> load_mnist_dataset_labels(const char* file_name);
}
