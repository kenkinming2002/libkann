#include <libkann/datasets/MNIST.hpp>

#include <libkann/Tensor.hpp>

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

  std::vector<Tensor<float>> load_mnist_dataset_images(const char* file_name)
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

    std::vector<Tensor<float>> images;
    images.reserve(count);

    for(uint32_t i=0; i<count; ++i)
    {
      Tensor<float> image = Tensor<float>::create(Shape{1, MNIST_DATASET_IMAGE_WIDTH, MNIST_DATASET_IMAGE_WIDTH});

      uint8_t image_data[MNIST_DATASET_IMAGE_WIDTH * MNIST_DATASET_IMAGE_WIDTH];
      if(!idx_file.file.read(reinterpret_cast<char*>(image_data), sizeof image_data))
        throw std::runtime_error("MNIST Data Set - Invalid file format");

      ranges::copy(image_data | ranges::views::transform([](uint8_t b) { return (float)b / 255; }), image.data());
      images.push_back(std::move(image));
    }

    return images;
  }

  std::vector<Tensor<float>> load_mnist_dataset_labels(const char* file_name)
  {
    IDXFile idx_file(file_name);
    if(idx_file.data_type != DataType::UNSIGNED_BYTE)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    if(idx_file.dimensions.size() != 1)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    uint32_t count = idx_file.dimensions[0];

    std::vector<Tensor<float>> labels;
    labels.reserve(count);

    for(uint32_t i=0; i<count; ++i)
    {
      Tensor<float> label = Tensor<float>::create(Shape{10});

      uint8_t label_data;
      if(!idx_file.file.read(reinterpret_cast<char*>(&label_data), sizeof label_data))
        throw std::runtime_error("MNIST Data Set - Invalid file format");

      ranges::copy(ranges::views::ints(0,10) | ranges::views::transform([label_data](auto i) -> float { return i == label_data ? 1.0 : 0.0; }), label.data());
      labels.push_back(std::move(label));
    }

    return labels;
  }
}
