#pragma once

#include "Config.hpp"
#include "Selectable.hpp"
#include "StaticBody.hpp"

#include <Eigen/Eigen>

#include <algorithm>

class BerryBush : public StaticBody, public Selectable
{
public:
  BerryBush(Eigen::Vector2d position);

public:
  friend class Renderer;

public:
  auto count() const { return m_berryCount; }
  void take(size_t count=1) { m_berryCount -= count; }

public:
  template<typename InputIterator>
  static void batchUpdate(InputIterator first, InputIterator last, float dt);

private:
  size_t m_berryCount;
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
