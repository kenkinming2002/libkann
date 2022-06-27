#pragma once

#include <libkann/Export.hpp>
#include <libkann/Tensor.hpp>

#include <vector>

namespace kann
{
  static constexpr size_t MNIST_DATASET_IMAGE_WIDTH = 28;
  KANN_EXPORT std::vector<Tensor<const float>> load_mnist_dataset_images(const char* file_name);
  KANN_EXPORT std::vector<Tensor<const float>> load_mnist_dataset_labels(const char* file_name);
}
