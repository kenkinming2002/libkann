#include "DataSet.hpp"

#include "IDXFile.hpp"

#include <cassert>
#include <iostream>
#include <limits>

namespace
{
  void showProgressBar(std::string_view name, size_t count, size_t total)
  {
    constexpr static size_t width = 40;
    std::cout << "\e[?25l";

    std::cout << name << "[";
    for(size_t i=0; i<width; ++i)
      if((float)i/width < (float)count/total)
        std::cout << "=";
      else
        std::cout << " ";

    std::cout << " - ";
    std::cout << count << "/" << total;
    std::cout << "]\r";

    std::cout << "\e[?25h";
  }
}

DataSet::DataSet(const char* imageFileName, const char* labelFileName)
{
  IDXFile imageFile(imageFileName), labelFile(labelFileName);

  assert(imageFile.type() == IDXFile::Type::UNSIGNED_BYTE);
  assert(labelFile.type() == IDXFile::Type::UNSIGNED_BYTE);

  assert(imageFile.dimensionCount() == 3);
  assert(labelFile.dimensionCount() == 1);

  assert(imageFile.dimensions()[0] == labelFile.dimensions()[0]);
  assert(imageFile.dimensions()[1] == Data::IMAGE_WIDTH);
  assert(imageFile.dimensions()[2] == Data::IMAGE_WIDTH);

  auto count = imageFile.dimensions()[0];
  m_data.resize(count);
  for(auto& data: m_data)
  {
    imageFile.read(reinterpret_cast<char*>(data.image.data()), data.image.size());
    labelFile.read(reinterpret_cast<char*>(&data.label), sizeof data.label);
  }
}

void DataSet::train(kann::NeuralNetwork& nn, float learningRate)
{
  Eigen::VectorXd expectedOutput(10);
  for(size_t i=0; i<m_data.size(); ++i)
  {
    showProgressBar("Tranining", i, m_data.size());

    const auto& data = m_data[i];

    Eigen::VectorXd input(data.image.size());
    for(size_t i=0; i<data.image.size(); ++i)
      input(i) = static_cast<double>(data.image[i])/255;

    nn.feedForward(std::move(input));

    expectedOutput = Eigen::VectorXd::Zero(10);
    expectedOutput(data.label) = 1.0;

    nn.backPropagate(expectedOutput);
    nn.train(learningRate);
  }

  std::cout << std::endl;
}

void DataSet::test(kann::NeuralNetwork& nn)
{
  size_t correct = 0;
  for(size_t i=0; i<m_data.size(); ++i)
  {
    showProgressBar("Testing", i, m_data.size());

    const auto& data = m_data[i];

    Eigen::VectorXd input(data.image.size());
    for(size_t i=0; i<data.image.size(); ++i)
      input(i) = static_cast<double>(data.image[i])/255;

    nn.feedForward(std::move(input));

    uint8_t label = std::numeric_limits<uint8_t>::max();;
    double record = -std::numeric_limits<double>::infinity();

    const auto& output = nn.output();
    for(size_t i=0; i<OUTPUT_LAYER_SIZE; ++i)
      if(output(i)>record)
      {
        label = i;
        record = output(i);
      }

    if(label == data.label)
      ++correct;
  }

  std::cout << std::endl;

  double correctionRate = 100.0 * static_cast<double>(correct) / m_data.size();
  std::cout << "Correction Rate:" << correctionRate << "%\n";
}
