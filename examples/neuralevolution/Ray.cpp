#include "Ray.hpp"

#include <cmath>
#include <limits>

#include <iostream>

double Ray::cast(CircleCollider circle) const
{
  Eigen::Vector2d offset = circle.position - position;

  Eigen::Vector2d rayTangent = Eigen::Rotation2Dd(angle) * Eigen::Vector2d(1.0, 0.0);
  Eigen::Vector2d rayNormal = Eigen::Vector2d(-rayTangent(1), rayTangent(0));

  double tangentProjectionLength = offset.dot(rayTangent);
  double normalProjectionLength = offset.dot(rayNormal);

  if(normalProjectionLength >= circle.radius)
    return std::numeric_limits<double>::infinity(); // We did not hit

  return tangentProjectionLength - std::sqrt(circle.radius * circle.radius - normalProjectionLength * normalProjectionLength);
}
