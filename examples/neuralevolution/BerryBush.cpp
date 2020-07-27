#include "BerryBush.hpp"

BerryBush::BerryBush(Eigen::Vector2d position) 
  : StaticBody(position, CONFIG.berryBush.radius), m_berryCount(CONFIG.berryBush.maxBerryCount) {}

float BerryBush::m_growth = 0.0f;
