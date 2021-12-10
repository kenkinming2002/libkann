#pragma once

#include "Entity.hpp"
#include "Renderer.hpp"

#include <box2d/box2d.h>

#include <Eigen/Eigen>

#include <algorithm>

class BerryBush : public Entity
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
  BerryBush(b2World& world, const Config& config, b2Vec2 position);

public:
  friend class Renderer;

public:
  void update(const Config& config, float dt);

public:
  void draw(const Config& config, Renderer& renderer) const;

public:
  auto count() const { return m_berryCount; }

  /* @return amount of energy taken */
  [[nodiscard]] double take(const Config& config, size_t count=1)
  {
    m_berryCount -= count;
    return config.energyPerBerry * count;
  }

private:
  size_t m_berryCount;
  float m_growth;
};
