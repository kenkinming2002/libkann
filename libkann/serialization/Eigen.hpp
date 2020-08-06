#pragma once

#include <Eigen/Eigen>

#include <cereal/cereal.hpp>
#include <cstddef>
#include <type_traits>

#include <iostream>

// Eigen Matrix
namespace cereal
{
  template<typename Archive, typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  void save(Archive& archive, const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& matrix)
  {
    archive(make_size_tag(matrix.size()), matrix.rows(), matrix.cols());

    for(size_t i=0; i<matrix.size(); ++i)
      archive(matrix.data()[i]);
  }

  template<typename Archive, typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  void load(Archive& archive, Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& matrix)
  {
    size_type size;
    Eigen::Index rows, cols;
    archive(make_size_tag(size), rows, cols);

    matrix.resize(rows, cols);
    for(size_t i=0; i<matrix.size(); ++i)
      archive(matrix.data()[i]);
  }
}
