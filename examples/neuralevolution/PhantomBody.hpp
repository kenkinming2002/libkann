#pragma once

#include "StaticBody.hpp"

#include <Eigen/Eigen>

class World;

/*
 * This is a class like RigidBody in many physics engine or game engine, but
 * without the collision component, this acting like a phantom. This is to
 * simplify the computation as our goal is not a realisitc game but just a
 * neurological simulation.
 *
 * This may change if we wanna investigate collision avoidance behavior.
 */
class PhantomBody : public StaticBody
{
public:
  PhantomBody(Eigen::Vector2d position, double radius);

public:
  double direction() const { return m_direction; }

public:
  void applyImpulse(double magnitude, double direction);
  void applyImpulseRelative(double magnitude, double relativeDirection);
  void applyImpulse(Eigen::Vector2d impulse);

public:
  void update(float dt, const World& world);

private:
  Eigen::Vector2d m_velocity;
  double m_direction;
};
