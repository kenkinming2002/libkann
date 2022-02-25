#include <algorithm>
#include <libkann/datasets/MNISTDataSet.hpp>

#include <stdexcept>
#include <iostream>

namespace kann
{
  MNISTDataSet::MNISTDataSet(const char* imageFileName, const char* labelFileName)
  {
    IDXFile imageFile(imageFileName);
    IDXFile labelFile(labelFileName);

    if(imageFile.type() != IDXFile::Type::UNSIGNED_BYTE)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    if(labelFile.type() != IDXFile::Type::UNSIGNED_BYTE)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    const auto& imageFileDimensions = imageFile.dimensions();
    if(imageFileDimensions.size() != 3)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    if(imageFileDimensions[1] != IMAGE_WIDTH)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    if(imageFileDimensions[2] != IMAGE_WIDTH)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    const auto& labelFileDimensions = labelFile.dimensions();
    if(labelFileDimensions.size() != 1)
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    if(imageFileDimensions[0] != labelFileDimensions[0])
      throw std::runtime_error("MNIST Data Set - Invalid file format");

    const size_t count = imageFileDimensions[0];
    for(size_t i=0; i<count; ++i)
    {
      Data data;
      if(!imageFile.read(reinterpret_cast<char*>(data.image), sizeof data.image))
        break;

      if(!labelFile.read(reinterpret_cast<char*>(&data.label), sizeof data.label))
        break;

      m_data.push_back(data);
    }
  }

  size_t MNISTDataSet::size() const
  {
    return m_data.size();
  }

  CRef<Tensor> MNISTDataSet::get(size_t column, size_t index) const
  {
    switch(column)
    {
    case COLUMN_IMAGE:
    {
      auto result = std::make_shared<Tensor>(IMAGE_SIZE);
      for(size_t i=0; i<IMAGE_SIZE; ++i)
        result->asArray()(i) = static_cast<double>(m_data[index].image[i])/255;
      return result;
    }
    case COLUMN_LABEL:
    {
      auto result = std::make_shared<Tensor>(10);
      for(uint8_t i=0; i<10; ++i)
        result->asArray()(i) = i == m_data[index].label ? 1.0 : 0.0;
      return result;
    }
    default:
      throw std::runtime_error("MNIST Data Set - get() - Invalid column");
    }
  }

  double MNISTDataSet::correctness(size_t column, size_t index, const Tensor& data) const
  {
    switch(column)
    {
    case COLUMN_IMAGE:
      throw std::runtime_error("MNIST Data Set - correctness() - Unsupported column COLUMN_IMAGE");
    case COLUMN_LABEL:
    {
      uint8_t label = -1;
      double record = -std::numeric_limits<double>::infinity();
      for(uint8_t i=0; i<data.size(); ++i)
        if(data.asArray()(i)>record)
        {
          label = i;
          record = data.asArray()(i);
        }

      return m_data[index].label == label ? 1.0 : 0.0;
    }
    default:
      throw std::runtime_error("MNIST Data Set - correctness() - Invalid column");
    }
  }
}
