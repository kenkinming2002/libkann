#include "BerryBush.hpp"

BerryBush::BerryBush(Config config, Eigen::Vector2d position)
  : StaticBody(position, config.radius),
    m_config(config),
    m_berryCount(config.maxBerryCount),
    m_growth(0.0f) {} // TODO: Add randomness to growth

void BerryBush::update(float dt)
{
  m_growth += dt * m_config.growthRate;
  if(m_growth>=1.0f)
  {
    m_growth-=1.0f;
    if(m_berryCount != m_config.maxBerryCount)
      ++m_berryCount;
  }
}
