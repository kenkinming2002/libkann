#pragma once

#include <libkann/export.hpp>

#include <Eigen/Eigen>

class Layer
{
public:
  Layer(size_t size) : m_input(size) {}

public:
  LIBKANN_SYMEXPORT Eigen::VectorXd& input() { return m_input; }
  LIBKANN_SYMEXPORT const Eigen::VectorXd& input() const { return m_input; }

  LIBKANN_SYMEXPORT Eigen::VectorXd output() const;

private:
  Eigen::VectorXd m_input;
};
