#include "World.hpp"

#include "Config.hpp"
#include "Generator.hpp"

#include <SFML/Graphics/RenderTarget.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>

static constexpr double AVERAGE_COUNT_PER_CELL = 3.0;

static double GRID_DIVISION_LENGTH_CREATURE()
{
  return std::sqrt((CONFIG.world.width * CONFIG.world.height) / (static_cast<double>(CONFIG.world.initialCreaturesCount) / AVERAGE_COUNT_PER_CELL));
}

static double GRID_DIVISION_LENGTH_BERRYBUSH()
{
  double initialBerryBushesCount = static_cast<double>(CONFIG.world.initialBerryBushesClusterCount) * static_cast<double>(CONFIG.world.initialBerryBushesClusterSizeMin+CONFIG.world.initialBerryBushesClusterSizeMin) / 2.0;
  return std::sqrt((CONFIG.world.width * CONFIG.world.height) / (initialBerryBushesCount / AVERAGE_COUNT_PER_CELL));
}

World::World(seed_type seed) 
  : m_dimension(CONFIG.world.width, CONFIG.world.height), 
    m_creatures(Grid<Creature>::centered_tag, {0.0, 0.0}, {CONFIG.world.width, CONFIG.world.height}, GRID_DIVISION_LENGTH_CREATURE()),
    m_berryBushes(Grid<BerryBush>::centered_tag, {0.0, 0.0}, {CONFIG.world.width, CONFIG.world.height}, GRID_DIVISION_LENGTH_BERRYBUSH()),
    m_generator(seed)
{
  generateClusters(m_berryBushes, m_generator, CONFIG.berryBush.radius, CONFIG.world.initialBerryBushesClusterCount, CONFIG.world.initialBerryBushesClusterSizeMin, CONFIG.world.initialBerryBushesClusterSizeMax, [this](StaticBody staticBody){
      m_berryBushes.emplace(staticBody.position(), staticBody.position());
  });

  generateNormal(m_creatures, m_generator, CONFIG.creature.radius, CONFIG.world.initialCreaturesCount, [this](StaticBody staticBody){
      m_creatures.emplace(staticBody.position(), m_generator, staticBody.position());
  });

}

void World::update(float dt)
{
  m_updateTimer.begin();

  m_worldTime += dt;

  auto berryBushes = m_berryBushes.all();
  BerryBush::batchUpdate(berryBushes.begin(), berryBushes.end(), dt);
  auto creatures = m_creatures.all();
  Creature::batchUpdate(creatures.begin(), creatures.end(), dt, *this);

  static constexpr auto positionFunc = [](const Creature& creature) { return creature.position(); };

  m_creatures.synchronize(positionFunc); // Register the updated position

  m_birthCount += m_newborns.size();
  m_deathToll += m_creatures.remove_if(std::mem_fn(&Creature::dead));
  m_creatures.insert(positionFunc, m_newborns.begin(), m_newborns.end());
  m_newborns.clear();

  m_updateTimer.end();
}

World::Info World::info() const
{
  auto creatures = m_creatures.all();

  Info info;

  info.ageStatistics = Statistics<float, float>(creatures.begin(), creatures.end(), [](const Creature& creature){
      return creature.statistics().lifetime;
  });

  info.matingCountStatistics = Statistics<float, size_t>(creatures.begin(), creatures.end(), [](const Creature& creature){
      return creature.statistics().matingCount;
  });

  info.healthyCreaturesCount =  std::count_if(creatures.begin(), creatures.end(), std::mem_fn(&Creature::healthy));
  info.creaturesCount = creatures.size();

  info.deathToll = m_deathToll;
  info.birthCount = m_birthCount;

  info.realTime = m_startTime.getElapsedTime().asSeconds();
  info.worldTime = m_worldTime;

  info.averageUpdateTime = m_updateTimer.average();

  return info;
}

World::result_variant World::find(Eigen::Vector2d position)
{
  std::variant<std::monostate, std::reference_wrapper<Creature>, std::reference_wrapper<BerryBush>> result = std::monostate{};
  Box queryBox(position, Eigen::Vector2d(0.0f, 0.0f));

  this->creatures().query(queryBox, [&](auto& creature){
    if((creature.position() - position).squaredNorm() < CONFIG.creature.radius * CONFIG.creature.radius)
      result = std::ref(creature);
  });
  this->berryBushes().query(queryBox, [&](auto& berryBush){
    if((berryBush.position() - position).squaredNorm() < CONFIG.berryBush.radius * CONFIG.berryBush.radius)
      result = std::ref(berryBush);
  });
  return result;
}
