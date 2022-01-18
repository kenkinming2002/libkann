#include <libkann/datasets/RandomDataSet.hpp>

#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace kann
{
  RandomDataSet::RandomDataSet(size_t dataSize, size_t size)
    : m_dataSize(dataSize), m_size(size) {}

  size_t RandomDataSet::size() const { return m_size; }

  Tensor RandomDataSet::get(size_t column, size_t index) const
  {
    if(column != COLUMN_DATA)
      throw std::runtime_error("Random Data Set - correctness() - Invalid column");

    Tensor result(m_dataSize);
    result.asArray().setRandom();
    return result;
  }

  double RandomDataSet::correctness(size_t column, size_t index, const Tensor& data) const
  {
    throw std::runtime_error("Random Data Set - correctness() - Invalid operation");
  }
}

