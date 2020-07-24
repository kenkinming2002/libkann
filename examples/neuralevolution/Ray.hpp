#pragma once

#include <Eigen/Eigen>

struct CircleCollider
{
public:
  CircleCollider(Eigen::Vector2d position, double radius)
    : position(position), radius(radius) {}


public:
  Eigen::Vector2d position;
  double radius;
};

struct Ray
{
public:
  Ray() = default;
  Ray(Eigen::Vector2d position, double angle);

public:
  double cast(CircleCollider circle) const;

public:
  Eigen::Vector2d position;
  double angle;

public:
  Eigen::Vector2d tangent, normal;
};
