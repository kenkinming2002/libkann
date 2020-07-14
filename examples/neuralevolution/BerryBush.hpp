#pragma once

#include <Eigen/Eigen>

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
  void update(float dt);

private:
  Eigen::Vector2d m_position;
  size_t m_berryCount;
  float m_growth;
};
