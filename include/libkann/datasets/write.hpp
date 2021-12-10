#pragma once

#include <Eigen/Eigen>

#include <ostream>

namespace kann
{
  void writeImage(std::ostream& os, const Eigen::VectorXd& data, size_t width, size_t height);
}
