#pragma once

#include <Eigen/Eigen>

#include <SFML/Graphics/Drawable.hpp>
#include <ostream>

class BerryBush : public sf::Drawable
{
public:
  BerryBush(Eigen::Vector2d position);

public:
  auto position() const { return m_position; }
  auto count() const { return m_berryCount; }
  void take(size_t count=1) { m_berryCount -= count; }

public:
  void update(float dt);

public:
  void draw(sf::RenderTarget &target, sf::RenderStates states) const override;

private:
  Eigen::Vector2d m_position;
  size_t m_berryCount;
  float m_growth;
};
