#pragma once

#include <libkann/neural_networks/NeuralNetwork.hpp>

#include <array>
#include <vector>

struct Data
{
public:
  static constexpr size_t IMAGE_WIDTH = 28;
  static constexpr size_t IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH;

public:
  std::array<std::byte, IMAGE_SIZE> image;
  uint8_t label;
};

class DataSet
{
public:
  static constexpr size_t INPUT_LAYER_SIZE = Data::IMAGE_SIZE;
  static constexpr size_t OUTPUT_LAYER_SIZE = 10;

public:
  DataSet(const char* imageFileName, const char* labelFileName);

public:
  void train(kann::NeuralNetwork& nn, float learningRate);
  void test(kann::NeuralNetwork& nn);

private:
  std::vector<Data> m_data;
};
