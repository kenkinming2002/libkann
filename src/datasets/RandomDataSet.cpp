#include <libkann/datasets/RandomDataSet.hpp>

#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace kann
{
  RandomDataSet::RandomDataSet(size_t dataSize, size_t size)
    : m_dataSize(dataSize), m_size(size) {}

  size_t RandomDataSet::size() const { return m_size; }

  void RandomDataSet::get(size_t column, size_t /*index*/, Eigen::VectorXd& data) const
  {
    if(column != COLUMN_DATA)
      throw std::runtime_error("Random Data Set - correctness() - Invalid column");

    data = Eigen::VectorXd::Random(m_dataSize);
  }

  double RandomDataSet::correctness(size_t column, size_t /*index*/, const Eigen::VectorXd& data) const
  {
    throw std::runtime_error("Random Data Set - correctness() - Invalid operation");
  }
}

