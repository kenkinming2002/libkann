#pragma once

#include "Selectable.hpp"
#include "StaticBody.hpp"

#include <Eigen/Eigen>

#include <algorithm>

class BerryBush : public StaticBody, public Selectable
{
public:
  struct Config
  {
    double energyPerBerry;
    size_t maxBerryCount;
    float growthRate;

    double radius;
  };

public:
  BerryBush(Config config, Eigen::Vector2d position);

public:
  friend class Renderer;

public:
  void update(float dt);

public:
  auto count() const { return m_berryCount; }

  /* @return amount of energy taken */
  [[nodiscard]] double take(size_t count=1)
  {
    m_berryCount -= count;
    return m_config.energyPerBerry * count;
  }

private:
  Config m_config;
  size_t m_berryCount;
  float m_growth;
};
