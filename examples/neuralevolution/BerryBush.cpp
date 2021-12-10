#include "BerryBush.hpp"

BerryBush::BerryBush(b2World& world, const Config& config, b2Vec2 position)
  : Entity(Entity::Type::BERRY_BUSH, world, position, config.radius)
{
  m_berryCount = config.maxBerryCount;
  m_growth = 0.0f;
}

void BerryBush::update(const Config& config, float dt)
{
  // TODO: Add randomness to growth
  m_growth += dt * config.growthRate;
  if(m_growth>=1.0f)
  {
    m_growth-=1.0f;
    if(m_berryCount != config.maxBerryCount)
      ++m_berryCount;
  }
}

void BerryBush::draw(const Config& config, Renderer& renderer) const
{
  const auto radius = this->radius();
  const auto position = this->position();

  const auto BAR_THICKNESS       = radius * 0.2f;
  const auto BAR_VERTICAL_OFFSET = radius * 1.2f;

  renderer.addCircle({position.x, position.y}, radius, sf::Color::Green);
  renderer.addBar({position.x, position.y+BAR_VERTICAL_OFFSET},
      {radius * 2.0f, BAR_THICKNESS}, sf::Color::Green, sf::Color::Red,
      (float)m_berryCount / config.maxBerryCount
  );
}
