#include "World.hpp"

#include "Config.hpp"
#include "Generator.hpp"

#include <SFML/Graphics/RenderTarget.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>

static constexpr double AVERAGE_COUNT_PER_CELL = 3.0;

namespace
{
  template<typename T>
  static Grid<T> createGrid(double width, double height, size_t count, size_t countPerCell)
  {
    double size       = width * height;
    double cellsCount = (double)count / countPerCell;
    double divisionLength = size / cellsCount;
    return Grid<T>(Grid<T>::centered_tag, {0.0, 0.0}, {width, height}, divisionLength);
  }
}

World::World(Config config)
  : m_dimension(config.width, config.height),
  m_creatures(createGrid<Creature>(config.width, config.height,
        config.initialCreaturesCount, AVERAGE_COUNT_PER_CELL)),
  m_berryBushes(createGrid<BerryBush>(config.width, config.height,
        config.initialBerryBushesClusterCount *
        config.initialBerryBushesClusterSizeMin+config.initialBerryBushesClusterSizeMin,
        AVERAGE_COUNT_PER_CELL)),
    m_generator(config.seed)
{
  generateClusters(m_berryBushes, m_generator, CONFIG.berryBush.radius,
      config.initialBerryBushesClusterCount,
      config.initialBerryBushesClusterSizeMin,
      config.initialBerryBushesClusterSizeMax, [this](StaticBody staticBody){
      m_berryBushes.emplace(staticBody.position(), staticBody.position());
  });

  generateNormal(m_creatures, m_generator, CONFIG.creature.radius,
      config.initialCreaturesCount, [this](StaticBody staticBody){
      m_creatures.emplace(staticBody.position(), m_generator,
          staticBody.position());
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

  for(auto&& newborn : m_newborns)
    m_creatures.insert(newborn.position(), std::move(newborn));

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
