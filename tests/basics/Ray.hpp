#pragma once

#include <Eigen/Eigen>

struct CircleCollider
{
  Eigen::Vector2d position;
  double radius;
};

struct Ray
{
  Eigen::Vector2d position;
  double angle;

  double cast(CircleCollider circle) const;
};
