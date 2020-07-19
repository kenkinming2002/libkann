#include "BerryBush.hpp"

BerryBush::BerryBush(Eigen::Vector2d position) 
  : m_position(position), m_berryCount(CONFIG.berryBush.maxBerryCount) {}

float BerryBush::m_growth = 0.0f;
