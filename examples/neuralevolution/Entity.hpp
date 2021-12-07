#pragma once

#include <box2d/box2d.h>
#include <memory>

class Entity
{
public:
  enum class Type
  {
    BERRY_BUSH,
    CREATURE
  };

public:
  Entity(Type type, b2World& world, b2Vec2 position, float radius);
  ~Entity();

public:
  Entity(Entity&& other);
  Entity& operator=(Entity&& other);

public:
  Entity(const Entity& other) = delete;
  Entity& operator=(const Entity& other) = delete;

public:
  Type type() const { return m_type; }

public:
  void applyForwardForce(float magnitude);
  void applyTorque(float magnitude);

public:
  // Angle is relative the entity rotation
  struct RaycastResult
  {
    Entity* entity;
    float distance;
  };
  bool raycast(float angle, float length, RaycastResult& result);

public:
  b2Vec2 position() const;
  float angle() const;
  float radius() const;

private:
  Type m_type;
  b2Body* m_body;
};
