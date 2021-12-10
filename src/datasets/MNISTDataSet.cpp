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

  void MNISTDataSet::get(size_t index, Eigen::VectorXd& input, Eigen::VectorXd& output) const
  {
    const auto& data = m_data[index];

    input.resize(IMAGE_SIZE);
    for(size_t i=0; i<IMAGE_SIZE; ++i)
      input(i) = static_cast<double>(data.image[i])/255;

    output.resize(10);
    for(uint8_t i=0; i<10; ++i)
      output(i) = i == data.label ? 1.0 : 0.0;
  }

  double MNISTDataSet::correctness(size_t index, const Eigen::VectorXd& output) const
  {
    const auto& data = m_data[index];

    uint8_t label = -1;
    double record = -std::numeric_limits<double>::infinity();

    for(uint8_t i=0; i<output.size(); ++i)
      if(output(i)>record)
      {
        label = i;
        record = output(i);
      }

    return data.label == label ? 1.0 : 0.0;
  }
}
