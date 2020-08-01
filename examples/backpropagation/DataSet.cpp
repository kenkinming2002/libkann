#include "DataSet.hpp"

#include "IDXFile.hpp"

#include <cassert>
#include <iostream>

DataSet::DataSet(const char* imageFileName, const char* labelFileName)
{
  IDXFile imageFile(imageFileName), labelFile(labelFileName);

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

void DataSet::train(NeuralNetwork& nn, float learningRate)
{
  for(const auto& data: m_data)
  {
    for(size_t i=0; i<data.image.size(); ++i)
      nn.input(i, static_cast<double>(data.image[i])/255);
    nn.feedForward();

    Eigen::VectorXd expectedOutput = Eigen::VectorXd::Zero(10);
    expectedOutput(data.label) = 1.0;

    nn.backPropagate(expectedOutput);
    nn.train(learningRate);
  }
}

void DataSet::test(NeuralNetwork& nn)
{
  size_t correct = 0;
  for(const auto& data: m_data)
  {
    for(size_t i=0; i<INPUT_LAYER_SIZE; ++i)
      nn.input(i, static_cast<double>(data.image[i])/255);

    nn.feedForward();

    uint8_t label;
    double record = -std::numeric_limits<double>::infinity();
    for(size_t i=0; i<OUTPUT_LAYER_SIZE; ++i)
      if(auto output = nn.output(i); output>record)
      {
        label = i;
        record = output;
      }

    if(label == data.label)
      ++correct;
  }
  double correctionRate = 100.0 * static_cast<double>(correct) / m_data.size();
  std::cout << "Correction Rate:" << correctionRate << "%\n";
}
