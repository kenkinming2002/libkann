#include "World.hpp"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RectangleShape.hpp>

#include <iostream>
#include <algorithm>

#include "Config.hpp"

#include <cmath>

static constexpr double AVERAGE_COUNT_PER_CELL = 10.0;

static double GRID_DIVISION_LENGTH_CREATURE()
{
  return std::sqrt((CONFIG.world.width * CONFIG.world.height) / (static_cast<double>(CONFIG.world.initialCreaturesCount) / 10.0));
}

static double GRID_DIVISION_LENGTH_BERRYBUSH()
{
  return std::sqrt((CONFIG.world.width * CONFIG.world.height) / (static_cast<double>(CONFIG.world.initialBerryBushesCount) / 10.0));
}

World::World(sf::View view, seed_type seed) 
  : m_defaultView(view), m_view(view), m_dimension(CONFIG.world.width, CONFIG.world.height), 
    m_creatures(Grid<Creature>::centerd_tag, {0.0, 0.0}, {CONFIG.world.width, CONFIG.world.height}, GRID_DIVISION_LENGTH_CREATURE()),
    m_berryBushes(Grid<BerryBush>::centerd_tag, {0.0, 0.0}, {CONFIG.world.width, CONFIG.world.height}, GRID_DIVISION_LENGTH_BERRYBUSH()),
    m_generator(seed)
{
  std::uniform_real_distribution<double> xPositionDistribution(-m_dimension(0)/2.0, m_dimension(0)/2.0);
  std::uniform_real_distribution<double> yPositionDistribution(-m_dimension(1)/2.0, m_dimension(1)/2.0);

  for(size_t i=0; i<CONFIG.world.initialCreaturesCount; ++i)
  {
    Eigen::Vector2d position(xPositionDistribution(m_generator), yPositionDistribution(m_generator));
    m_creatures.emplace(position, m_generator, position);
  }

  for(size_t i=0; i<CONFIG.world.initialBerryBushesCount; ++i)
  {
    Eigen::Vector2d position(xPositionDistribution(m_generator), yPositionDistribution(m_generator));
    m_berryBushes.emplace(position, position);
  }
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
#pragma omp parallel for
  for(std::reference_wrapper<BerryBush> berryBush: m_berryBushes.all())
    berryBush.get().update(dt);

#pragma omp parallel for
  for(std::reference_wrapper<Creature> creature: m_creatures.all())
    creature.get().preUpdate(dt, *this);

  for(std::reference_wrapper<Creature> creature: m_creatures.all())
    creature.get().update(dt, *this);

  m_creatures.synchronize(std::mem_fn(&Creature::position)); // Register the updated position

  m_birthCount += m_newborns.size();
  m_deathToll += m_creatures.remove_if(std::mem_fn(&Creature::dead));
  m_creatures.insert(std::mem_fn(&Creature::position), m_newborns.begin(), m_newborns.end());
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

  for(std::reference_wrapper<const BerryBush> berryBush: m_berryBushes.all())
    target.draw(berryBush, states);

  for(std::reference_wrapper<const Creature> creature: m_creatures.all())
    target.draw(creature, states);
}

void World::log() const
{
  auto creatures = m_creatures.all();

  size_t healthyCreaturesCount =  std::count_if(creatures.begin(), creatures.end(), std::mem_fn(&Creature::healthy));
  size_t creaturesCount = creatures.size();

  std::cout << "Creatures:" << healthyCreaturesCount << "/" << creaturesCount << "(Healthy/All)" << ", " 
    << m_deathToll << "/" << m_birthCount << "(DeathToll/BirthCount)";
}
