#include "Entity.hpp"

#include <limits>
#include <iostream>

Entity::Entity(Type type, b2World& world, b2Vec2 position, float radius)
  : m_type(type)
{
  b2BodyDef bodyDef;
  bodyDef.position = position;
  bodyDef.type     = b2_dynamicBody;

  m_body = world.CreateBody(&bodyDef); // Somehow we cannot assign directly
  m_body->GetUserData().pointer = reinterpret_cast<uintptr_t>(this); // This may become dangling

  b2CircleShape shape;
  shape.m_p      = b2Vec2(0.0f, 0.0f);
  shape.m_radius = radius;

  b2FixtureDef fixtureDef;
  fixtureDef.shape = &shape;

  // TODO: Unhardcode this value
  fixtureDef.density  = 1.0f;
  fixtureDef.friction = 0.3f;

  m_body->CreateFixture(&fixtureDef);
}

Entity::~Entity()
{
  if(m_body)
  {
    b2World* world = m_body->GetWorld();
    world->DestroyBody(m_body);
    m_body = nullptr;
  }
}

Entity::Entity(Entity&& other)
  : m_type(other.m_type), m_body(std::exchange(other.m_body, nullptr))
{
  m_body->GetUserData().pointer = reinterpret_cast<uintptr_t>(this);
}

Entity& Entity::operator=(Entity&& other)
{
  std::swap(m_type, other.m_type);
  std::swap(m_body, other.m_body);

  if(m_body)
    m_body->GetUserData().pointer = reinterpret_cast<uintptr_t>(this);

  if(other.m_body)
    other.m_body->GetUserData().pointer = reinterpret_cast<uintptr_t>(this);

  return *this;
}

void Entity::applyForwardForce(float magnitude)
{
  assert(m_body);
  float angle = this->angle();
  b2Vec2 force = b2Vec2(magnitude * std::cos(angle), magnitude * std::sin(angle));
  m_body->ApplyForceToCenter(force, true);
}

void Entity::applyTorque(float magnitude)
{
  assert(m_body);
  m_body->ApplyTorque(magnitude, false);
}

namespace
{
  struct RaycastCallback : public b2RayCastCallback
  {
  public:
    RaycastCallback(b2Vec2 initialPoint) : initialPoint(initialPoint) {}

  public:
    float ReportFixture(b2Fixture *fixture, const b2Vec2 &point, const b2Vec2 &normal, float fraction) override
    {
      b2Vec2 distance = initialPoint;
      distance *= -1.0f;
      distance += point;
      if(distance.LengthSquared() < closestDistance * closestDistance)
      {
        closestDistance = distance.Length();
        closestFixture  = fixture;
      }
      return 1;
    }

  public:
    const b2Vec2 initialPoint;

  public:
    float closestDistance = std::numeric_limits<float>::infinity();
    b2Fixture* closestFixture = nullptr;
  };
}

bool Entity::raycast(float angle, float length, RaycastResult& result)
{
  assert(m_body);

  b2RayCastOutput output;

  float realAngle = this->angle()+angle;
  b2Vec2 offset = b2Vec2(std::cos(realAngle), std::sin(realAngle));
  offset *= length;

  b2Vec2 point1 = this->position();
  b2Vec2 point2 = point1;
  point2 += offset;

  RaycastCallback callback(point1);
  m_body->GetWorld()->RayCast(&callback, point1, point2);
  if(!callback.closestFixture)
    return false;

  result.entity = reinterpret_cast<Entity*>(callback.closestFixture->GetBody()->GetUserData().pointer);
  result.distance = callback.closestDistance;
  return true;
}

b2Vec2 Entity::position() const
{
  assert(m_body);
  return m_body->GetPosition();
}

float Entity::angle() const
{
  assert(m_body);
  return m_body->GetAngle();
}

float Entity::radius() const
{
  assert(m_body);
  return static_cast<b2CircleShape*>(m_body->GetFixtureList()->GetShape())->m_radius;
}
