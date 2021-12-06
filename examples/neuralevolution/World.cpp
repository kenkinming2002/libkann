#include "World.hpp"

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

  static Grid<Creature> createCreatureGrid(const World::Config& config)
  {
    return createGrid<Creature>(config.width, config.height, config.initialCreaturesCount, AVERAGE_COUNT_PER_CELL);
  }

  static Grid<BerryBush> createBerryBushGrid(const World::Config& config)
  {
    auto initialBerryBushesCount = config.initialBerryBushesClusterCount * (config.initialBerryBushesClusterSizeMin+config.initialBerryBushesClusterSizeMax) / 2.0;
    return createGrid<BerryBush>(config.width, config.height, initialBerryBushesCount, AVERAGE_COUNT_PER_CELL);
  }
}

World::World(Config config, Creature::Config creatureConfig, Creature::NeuralNetworkConfig creatureNeuralNetworkConfig, BerryBush::Config berryBushConfig)
  : m_config(config), m_creatureConfig(creatureConfig), m_berryBushConfig(berryBushConfig),
    m_creatures(createCreatureGrid(m_config)),
    m_berryBushes(createBerryBushGrid(m_config)),
    m_generator(config.seed)
{
  generateClusters(m_berryBushes, m_generator, m_berryBushConfig.radius,
      m_config.initialBerryBushesClusterCount,
      m_config.initialBerryBushesClusterSizeMin,
      m_config.initialBerryBushesClusterSizeMax, [&](StaticBody staticBody){
      m_berryBushes.emplace(staticBody.position(),
          m_berryBushConfig,
          staticBody.position()
      );
  });

  generateNormal(m_creatures, m_generator, creatureConfig.maxRadius,
      config.initialCreaturesCount, [&](StaticBody staticBody){
      auto neuralNetwork = Creature::makeNeuralNetork(creatureNeuralNetworkConfig, m_generator);
      m_creatures.emplace(staticBody.position(),
          creatureConfig,
          std::move(neuralNetwork),
          staticBody.position(),
          creatureConfig.maxEnergy,
          creatureConfig.maxHealth
      );
  });
}

void World::update(float dt)
{
  m_updateTimer.begin();

  m_worldTime += dt;

  for(auto& berryBush : m_berryBushes.all())
    berryBush.get().update(dt);

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
    if((creature.position() - position).squaredNorm() < creature.radius() * creature.radius())
      result = std::ref(creature);
  });
  this->berryBushes().query(queryBox, [&](auto& berryBush){
    if((berryBush.position() - position).squaredNorm() < berryBush.radius() * berryBush.radius())
      result = std::ref(berryBush);
  });
  return result;
}
