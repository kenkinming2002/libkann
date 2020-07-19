#pragma once

#include "Config.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <ostream>

class BerryBush
{
public:
  BerryBush(Eigen::Vector2d position);

public:
  friend class Renderer;

public:
  auto position() const { return m_position; }
  auto count() const { return m_berryCount; }
  void take(size_t count=1) { m_berryCount -= count; }

public:
  template<typename InputIterator>
  static void batchUpdate(InputIterator first, InputIterator last, float dt);
  void update(float dt);

private:
  Eigen::Vector2d m_position;
  size_t m_berryCount;

private:
  static float m_growth;
};

template<typename InputIterator>
void BerryBush::batchUpdate(InputIterator first, InputIterator last, float dt)
{
  m_growth += dt * CONFIG.berryBush.growthRate;
  if(m_growth>=1.0f)
  {
    m_growth-=1.0f;
    std::for_each(first, last, [&](BerryBush& berryBush){ berryBush.m_berryCount = std::min(CONFIG.berryBush.maxBerryCount, berryBush.m_berryCount+1);} );
  }
}
