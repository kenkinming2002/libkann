#include <libkann/datasets/MNIST.hpp>

#include <libtensor/Tensor.hpp>

// TODO: Where do we find this header on windows?
#include <arpa/inet.h>

#include <range/v3/all.hpp>

#include <stdexcept>
#include <fstream>

namespace kann
{
  enum class DataType : uint8_t {
    UNSIGNED_BYTE = 0x08,
    SIGNED_BYTE = 0x09,
    SHORT = 0x0B,
    INT = 0x0C,
    FLOAT = 0x0D,
    DOUBLE = 0x0E
  };

  struct [[gnu::packed]] IDXFileHeader
  {
    char magic[2];
    DataType data_type;
    uint8_t dimensions_count;
  };
  static_assert(sizeof(IDXFileHeader) == 4);

  struct IDXFile
  {
    std::ifstream file;
    DataType data_type;
    std::vector<uint32_t> dimensions;

    IDXFile(const char* file_name)
    {
      file.open(file_name);
      if(!file)
        throw std::runtime_error(std::string("Failed to open file ") + file_name);

      IDXFileHeader header;
      if(!file.read(reinterpret_cast<char*>(&header), sizeof header))
        throw std::runtime_error(std::string("Invalid file header ") + file_name);

      if(header.magic[0] != 0 || header.magic[1] != 0)
        throw std::runtime_error(std::string("Invalid file format ") + file_name);

      data_type = header.data_type;
      dimensions.resize(header.dimensions_count);

      if(!file.read(reinterpret_cast<char*>(dimensions.data()), dimensions.size() * sizeof dimensions[0]))
        throw std::runtime_error(std::string("Invalid file format ") + file_name);

      for(auto& dimension: dimensions)
        dimension = ntohl(dimension);
    }
  };

  std::vector<tensor::Tensor<float>> load_mnist_dataset_images(const char* file_name)
  {
    IDXFile idx_file(file_name);
    if(idx_file.data_type != DataType::UNSIGNED_BYTE)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    if(idx_file.dimensions.size() != 3)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    if(idx_file.dimensions[1] != MNIST_DATASET_IMAGE_WIDTH ||
       idx_file.dimensions[2] != MNIST_DATASET_IMAGE_WIDTH)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    uint32_t count = idx_file.dimensions[0];

    std::vector<tensor::Tensor<float>> images;
    images.reserve(count);

    for(uint32_t i=0; i<count; ++i)
    {
      uint8_t image_data[MNIST_DATASET_IMAGE_WIDTH * MNIST_DATASET_IMAGE_WIDTH];
      if(!idx_file.file.read(reinterpret_cast<char*>(image_data), sizeof image_data))
        throw std::runtime_error("MNIST Data Set - Invalid file format");

      auto image_buffer = std::make_shared<tensor::Buffer<float>>(MNIST_DATASET_IMAGE_WIDTH * MNIST_DATASET_IMAGE_WIDTH);
      for(size_t i=0; i<MNIST_DATASET_IMAGE_WIDTH * MNIST_DATASET_IMAGE_WIDTH; ++i)
        image_buffer->data().data()[i] = (float)image_data[i] / 255.0f;

      images.push_back(tensor::Tensor<float>(tensor::Shape::make(1, MNIST_DATASET_IMAGE_WIDTH, MNIST_DATASET_IMAGE_WIDTH), std::move(image_buffer)));
    }

    return images;
  }

  std::vector<tensor::Tensor<float>> load_mnist_dataset_labels(const char* file_name)
  {
    IDXFile idx_file(file_name);
    if(idx_file.data_type != DataType::UNSIGNED_BYTE)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    if(idx_file.dimensions.size() != 1)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    uint32_t count = idx_file.dimensions[0];

    std::vector<tensor::Tensor<float>> labels;
    labels.reserve(count);

    for(uint32_t i=0; i<count; ++i)
    {
      uint8_t label_data;
      if(!idx_file.file.read(reinterpret_cast<char*>(&label_data), sizeof label_data))
        throw std::runtime_error("MNIST Data Set - Invalid file format");

      auto label_buffer = std::make_shared<tensor::Buffer<float>>(10);
      for(size_t i=0; i<10; ++i)
        label_buffer->data().data()[i] = label_data == i ? 1.0f : 0.0f;

      labels.push_back(tensor::Tensor<float>(tensor::Shape::make(10), std::move(label_buffer)));
    }

    return labels;
  }

  tensor::Tensor<float> load_mnist_dataset_images_single(const char* file_name)
  {
    IDXFile idx_file(file_name);
    if(idx_file.data_type != DataType::UNSIGNED_BYTE)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    size_t size = 1;
    for(uint32_t dimension : idx_file.dimensions)
      size *= dimension;

    auto data = std::make_unique_for_overwrite<uint8_t[]>(size);
    idx_file.file.read(reinterpret_cast<char*>(&data[0]), size * sizeof data[0]);

    auto buffer = std::make_shared<tensor::Buffer<float>>(size);
    for(size_t i=0; i<size; ++i)
      (*buffer)[i]  = (float)data[i] / 255.0f;

    tensor::Shape shape;
    shape.dimensions.reserve(idx_file.dimensions.size());
    for(uint32_t dimension : idx_file.dimensions)
      shape.dimensions.push_back(dimension);

    return tensor::Tensor<float>(std::move(shape), std::move(buffer));
  }

  tensor::Tensor<float> load_mnist_dataset_labels_single(const char* file_name)
  {
    IDXFile idx_file(file_name);
    if(idx_file.data_type != DataType::UNSIGNED_BYTE)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    size_t size = 1;
    for(uint32_t dimension : idx_file.dimensions)
      size *= dimension;

    auto data = std::make_unique_for_overwrite<uint8_t[]>(size);
    idx_file.file.read(reinterpret_cast<char*>(&data[0]), size * sizeof data[0]);

    auto buffer = std::make_shared<tensor::Buffer<float>>(size * 10);
    for(size_t i=0; i<size; ++i)
      for(size_t j=0; j<10; ++j)
        (*buffer)[i*10+j] = (data[i] == j) ? 1.0f : 0.0f;

    tensor::Shape shape;
    shape.dimensions.reserve(idx_file.dimensions.size());
    for(uint32_t dimension : idx_file.dimensions)
      shape.dimensions.push_back(dimension);
    shape.dimensions.push_back(10);

    return tensor::Tensor<float>(std::move(shape), std::move(buffer));
  }
}
