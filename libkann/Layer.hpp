#pragma once

#include <libkann/export.hpp>

#include <Eigen/Eigen>

class Layer
{
public:
  Layer(size_t size) : m_input(size) {}

public:
  LIBKANN_SYMEXPORT void input(Eigen::VectorXd input);
  LIBKANN_SYMEXPORT Eigen::VectorXd output() const;

  LIBKANN_SYMEXPORT friend std::ostream& operator<<(std::ostream& os, const Layer& layer);

private:
  Eigen::VectorXd m_input;
};
