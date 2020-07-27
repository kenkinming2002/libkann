#include "PhantomBody.hpp"

#include "World.hpp"

#include <cmath>
#include <cassert>

namespace
{
  constexpr double wrap(double v, double a, double b)
  {
    double gap = b - a;
    int n = std::floor((v-a) / gap);
    return  v - n * gap;
  }

  Eigen::Vector2d wrap(const Eigen::Vector2d& v, const Eigen::Vector2d& a, const Eigen::Vector2d& b)
  {
    return Eigen::Vector2d(wrap(v(0), a(0), b(0)), wrap(v(1), a(1), b(1)));
  }
}

PhantomBody::PhantomBody(Eigen::Vector2d position, double radius)
  : m_position(position), m_radius(radius), m_velocity(0.0, 0.0), m_direction(0.0) {}

void PhantomBody::applyImpulse(double magnitude, double direction)
{
  this->applyImpulse(magnitude * Eigen::Vector2d(std::cos(direction), std::sin(direction)));
}

void PhantomBody::applyImpulseRelative(double magnitude, double relativeDirection)
{
  this->applyImpulse(magnitude * (Eigen::Rotation2Dd(relativeDirection) * m_velocity.normalized()));
}

void PhantomBody::applyImpulse(Eigen::Vector2d impulse)
{
  m_velocity += impulse;
  m_direction = std::atan2(m_velocity(1), m_velocity(0));
}

void PhantomBody::update(float dt, const World& world)
{
  static constexpr double DRAG_COEFFICIENT = 0.001;
  static constexpr double FRICTION_COEFFICIENT = 0.01;

  if(double squaredNorm = m_velocity.squaredNorm(); squaredNorm != 0.0)
  {
    double norm = std::sqrt(squaredNorm);
    Eigen::Vector2d normalizedVelocity = m_velocity / norm;

    double drag = DRAG_COEFFICIENT * squaredNorm;
    double friction = FRICTION_COEFFICIENT * norm;
    double newMagnitude = std::max(norm - drag - friction, 0.0);
    m_velocity = normalizedVelocity * newMagnitude;

    m_position += dt * m_velocity;
  }

  Eigen::Vector2d halfWorldDimension = world.dimension() / 2.0;
  m_position = wrap(m_position, -halfWorldDimension, halfWorldDimension);
}
