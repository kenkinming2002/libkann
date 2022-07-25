#include <libtensor/SL.hpp>

#include <fstream>

namespace tensor
{
  namespace
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

    template<typename T, typename U>
    T numeric_cast(U value)
    {
      if(!std::in_range<T>(value))
        throw std::runtime_error("Overflow during integer conversion");

      return static_cast<T>(value);
    }

    template<typename T, typename U>
    std::vector<T> numeric_casts(const std::vector<U>& values)
    {
      std::vector<T> results;
      results.reserve(values.size());
      for(const auto value : values)
        results.push_back(numeric_cast<T>(value));

      return results;
    }

  }

  void save_tensor(tensor::Tensor<float> value, const std::string& filename)
  {
    std::ofstream file;
    file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    file.open(filename);

    // Header
    const auto& _dimensions_count = value.shape.dimensions.size();
    const auto& dimensions_count = numeric_cast<uint8_t>(_dimensions_count);

    IDXFileHeader header;
    header.magic[0] = {};
    header.magic[1] = {};
    header.data_type = DataType::FLOAT;
    header.dimensions_count = dimensions_count;
    file.write(reinterpret_cast<const char*>(&header), sizeof header);

    // Dimensions
    const auto& _dimensions       = value.shape.dimensions;
    const auto& dimensions       = numeric_casts<uint32_t>(_dimensions);
    file.write(reinterpret_cast<const char*>(dimensions.data()), dimensions.size() * sizeof dimensions[0]);

    // Buffer
    const auto& buffer           = value.buffer;
    file.write(reinterpret_cast<const char*>(buffer->data().data()), buffer->data().size_bytes());
  }

  tensor::Tensor<float> load_tensor(const std::string& filename)
  {
    std::ifstream file;
    file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    file.open(filename);

    // Header
    IDXFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof header);
    assert(header.magic[0] == 0);
    assert(header.magic[1] == 0);

    auto _dimensions_count = header.dimensions_count;
    auto dimensions_count  = numeric_cast<size_t>(_dimensions_count);

    // Dimensions
    auto _dimensions = std::vector<uint32_t>(dimensions_count);
    file.read(reinterpret_cast<char*>(_dimensions.data()), _dimensions.size() * sizeof _dimensions[0]);
    auto dimensions = numeric_casts<size_t>(_dimensions);

    // Buffer
    auto buffer = std::make_shared<tensor::Buffer<float>>(std::accumulate(dimensions.begin(), dimensions.end(), (size_t)1, std::multiplies<size_t>()));
    file.read(reinterpret_cast<char*>(buffer->data().data()), buffer->data().size_bytes());

    auto shape = tensor::Shape::from_vector(dimensions);
    auto tensor = tensor::Tensor<float>(std::move(shape), std::move(buffer));
    return tensor;
  }
}
