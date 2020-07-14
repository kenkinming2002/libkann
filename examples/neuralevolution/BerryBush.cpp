#include "BerryBush.hpp"

#include <iostream>
#include <cmath>

#include "Config.hpp"

BerryBush::BerryBush(Eigen::Vector2d position) 
  : m_position(position), m_berryCount(CONFIG.berryBush.maxBerryCount), m_growth(0) {}

void BerryBush::update(float dt)
{
  m_growth += dt * CONFIG.berryBush.growthRate;;
  if(m_growth>=1.0f)
  {
    m_growth-=1.0f;
    if(m_berryCount != CONFIG.berryBush.maxBerryCount)
      ++m_berryCount;
  }
}
