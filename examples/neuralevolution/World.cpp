#include "World.hpp"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RectangleShape.hpp>

#include <iostream>
#include <iomanip>
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

World::World(seed_type seed) 
  : m_dimension(CONFIG.world.width, CONFIG.world.height), 
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

void World::update(float dt)
{
  m_worldTime += dt;

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

void World::log() const
{
  auto creatures = m_creatures.all();

  size_t healthyCreaturesCount =  std::count_if(creatures.begin(), creatures.end(), std::mem_fn(&Creature::healthy));
  size_t creaturesCount = creatures.size();

  float realTime = m_startTime.getElapsedTime().asSeconds();

  std::cout << "Creatures:" << healthyCreaturesCount << "/" << creaturesCount << "(Healthy/All)" << ", " << m_deathToll << "/" << m_birthCount << "(DeathToll/BirthCount)" << "; " << "Time:" << realTime << "/" << m_worldTime << "/" << m_worldTime / realTime << "(Real/World/Ratio)";
}
