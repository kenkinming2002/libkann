#pragma once

#include <Eigen/Eigen>

class StaticBody
{
public:
  StaticBody(Eigen::Vector2d position, double radius);


public:
  Eigen::Vector2d position() const { return m_position; }
  const double& radius() const { return m_radius; }
  double& radius() { return m_radius; }

private:
  Eigen::Vector2d m_position;
  double m_radius;
};
