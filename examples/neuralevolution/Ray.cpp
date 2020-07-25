#include "Ray.hpp"

#include <tuple>
#include <utility>
#include <cmath>
#include <limits>
#include <iostream>

namespace
{
  std::pair<Eigen::Vector2d, Eigen::Vector2d> unitVectors(double angle)
  {
    double sin = std::sin(angle);
    double cos = std::cos(angle);
    return std::make_pair(Eigen::Vector2d(cos, sin), Eigen::Vector2d(-sin, cos));
  }
}

Ray::Ray(Eigen::Vector2d position, double angle) 
  : position(position), angle(angle) 
{
  std::tie(tangent, normal) = unitVectors(angle);
}

double Ray::cast(CircleCollider circle) const
{
  Eigen::Vector2d offset = circle.position - position;

  double tangentProjectionLength = offset.dot(this->tangent);
  double normalProjectionLength = offset.dot(this->normal);

  if(normalProjectionLength >= circle.radius)
    return std::numeric_limits<double>::infinity(); // We did not hit

  return tangentProjectionLength - 
    std::sqrt(circle.radius * circle.radius - normalProjectionLength * normalProjectionLength);
}
