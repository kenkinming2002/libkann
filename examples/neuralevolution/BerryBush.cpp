#include "BerryBush.hpp"

BerryBush::BerryBush(b2World& world, Config config, b2Vec2 position)
  : Entity(Entity::Type::BERRY_BUSH, world, position, config.radius), m_config(config)
{
  m_berryCount = config.maxBerryCount;
  m_growth = 0.0f;
}

void BerryBush::update(float dt)
{
  // TODO: Add randomness to growth
  m_growth += dt * m_config.growthRate;
  if(m_growth>=1.0f)
  {
    m_growth-=1.0f;
    if(m_berryCount != m_config.maxBerryCount)
      ++m_berryCount;
  }
}
