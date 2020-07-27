#pragma once

#include <Eigen/Eigen>

#include "StaticBody.hpp"
#include "PhantomBody.hpp"

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
  double cast(const StaticBody& staticBody) const;
  double cast(const PhantomBody& phantomBody) const;

public:
  Eigen::Vector2d position;
  double angle;

public:
  Eigen::Vector2d tangent, normal;
};
