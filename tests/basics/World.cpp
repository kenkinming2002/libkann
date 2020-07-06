#include "World.hpp"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RectangleShape.hpp>

#include <iostream>
#include <algorithm>

#include "Config.hpp"

World::World(sf::View view, seed_type seed) 
  : m_defaultView(view), m_view(view), m_dimension(CONFIG.world.width, CONFIG.world.height), m_generator(seed)
{
  std::uniform_real_distribution<double> xPositionDistribution(-m_dimension(0)/2.0, m_dimension(0)/2.0);
  std::uniform_real_distribution<double> yPositionDistribution(-m_dimension(1)/2.0, m_dimension(1)/2.0);

  for(size_t i=0; i<CONFIG.world.initialCreaturesCount; ++i)
    m_creatures.emplace_back(m_generator,
      Eigen::Vector2d(
        xPositionDistribution(m_generator), 
        yPositionDistribution(m_generator)
      )
    );

  for(size_t i=0; i<CONFIG.world.initialBerryBushesCount; ++i)
    m_berryBushes.emplace_back(Eigen::Vector2d(
        xPositionDistribution(m_generator), 
        yPositionDistribution(m_generator)
      )
    );
}

bool World::handleInput(sf::Event event)
{
  auto zoom = [this](float amount){
    m_view.zoom(amount);
    m_scale *= amount;
    return true;
  };
  auto move = [this](float x, float y){
    m_view.move(x * m_scale, y * m_scale);
    return true;
  };
  auto reset = [this](){
    m_view = m_defaultView;
    return true;
  };

  switch(event.type) {
    case sf::Event::KeyPressed:
      switch(event.key.code)
      {
        // Zoom in / Zoom out
        case sf::Keyboard::Add:
          return zoom(1.0f / ZOOM_SPEED);
        case sf::Keyboard::Subtract:
          return zoom(ZOOM_SPEED);

          // Up/Down/Left/Right
        case sf::Keyboard::H:
        case sf::Keyboard::Left:
          return move(-MOVE_SPEED, 0.0f);
        case sf::Keyboard::J:
        case sf::Keyboard::Down:
          return move(0.0f, MOVE_SPEED);
        case sf::Keyboard::K:
        case sf::Keyboard::Up:
          return move(0.0f, -MOVE_SPEED);
        case sf::Keyboard::L:
        case sf::Keyboard::Right:
          return move(MOVE_SPEED, 0.0f);

          // Reset
        case sf::Keyboard::Equal:
          return reset();
        default:
          return false;
      }
    // Zoom in / Zoom out
    case sf::Event::MouseWheelScrolled:
      if(event.mouseWheelScroll.delta >= 0.0f)
        return zoom(event.mouseWheelScroll.delta * ZOOM_SPEED);
      else
        return zoom(1.0f / (event.mouseWheelScroll.delta * ZOOM_SPEED));
      break;
    default:
      return false;
  }
}

void World::update(float dt)
{
  for(auto& berryBush: m_berryBushes)
    berryBush.update(dt);

#pragma omp parallel for
  for(auto& creature: m_creatures)
    creature.preUpdate(dt, *this);

  for(auto& creature: m_creatures)
    creature.update(dt, *this);

  m_creatures.erase(std::remove_if(m_creatures.begin(), m_creatures.end(), [](const auto& creature){ return creature.dead(); }), m_creatures.end());

  m_creatures.insert(m_creatures.end(), m_newborns.begin(), m_newborns.end());
  m_newborns.clear();
}

void World::draw(sf::RenderTarget &target, sf::RenderStates states) const
{
  target.setView(m_view);

  sf::RectangleShape rectangleShape;

  rectangleShape.setPosition(0.0f, 0.0f);
  rectangleShape.setSize({static_cast<float>(m_dimension(0)), static_cast<float>(m_dimension(1))});
  rectangleShape.setOrigin(m_dimension(0)/2.0, m_dimension(1)/2.0);

  rectangleShape.setFillColor(sf::Color::White);

  target.draw(rectangleShape, states);

  for(const auto& berryBush: m_berryBushes)
    target.draw(berryBush, states);

  for(const auto& creature: m_creatures)
    target.draw(creature, states);
}

void World::log() const
{
  std::cout << "Creature count:" << m_creatures.size() << '\n';
  std::cout << "Healthy Creature count:" << std::count_if(m_creatures.begin(), m_creatures.end(), [](const auto& creature){
      return creature.health() == CONFIG.creature.maxHealth;
  }) << '\n';
}
