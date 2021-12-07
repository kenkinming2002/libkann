#include "World.hpp"
#include "libkann/NeuralNetwork.hpp"
#include "libkann/RecurrentNeuralNetwork.hpp"

#include <SFML/Graphics/RenderTarget.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <utility>

World::World(Config config, Creature::Config creatureConfig, Creature::NeuralNetworkConfig creatureNeuralNetworkConfig, BerryBush::Config berryBushConfig)
  : m_config(config), m_creatureConfig(creatureConfig), m_berryBushConfig(berryBushConfig),
    m_generator(config.seed),
    m_world(b2Vec2(0.0f,0.0f)) // No gravity
{
  // Statistics

  // Create walls
  {
    /* Somehow we cannot assign directly, even if our custom deleter has default
     * constructor */

    const float wallThickness = 5.0f;
    const b2Vec2 positions[] = {
      b2Vec2(-(config.width+wallThickness)/2.0f, 0.0f                               ),
      b2Vec2( (config.width+wallThickness)/2.0f, 0.0f                               ),
      b2Vec2(0.0f                              , -(config.height+wallThickness)/2.0f),
      b2Vec2(0.0f                              ,  (config.height+wallThickness)/2.0f)
    };

    const b2Vec2 dimensions[] = {
      b2Vec2(wallThickness, config.height),
      b2Vec2(wallThickness, config.height),
      b2Vec2(config.width , wallThickness),
      b2Vec2(config.width , wallThickness)
    };

    for(size_t i=0; i<4; ++i)
    {
      const auto& position  = positions[i];
      const auto& dimension = dimensions[i];

      b2BodyDef bodyDef;
      bodyDef.position = position;
      bodyDef.type     = b2_staticBody;

      b2Body* body = m_world.CreateBody(&bodyDef);

      b2PolygonShape boxShape;
      boxShape.SetAsBox(dimension.x, dimension.y);

      b2FixtureDef fixtureDef;
      fixtureDef.shape = &boxShape;
      fixtureDef.density  = 1.0f;
      fixtureDef.friction = 0.3f;

      body->CreateFixture(&fixtureDef);
    }
  }

  // Creatures
  {
    std::uniform_real_distribution<double> distX(-m_config.width/2.0f,  m_config.width/2.0f);
    std::uniform_real_distribution<double> distY(-m_config.height/2.0f, m_config.height/2.0f);

    for(size_t i=0; i<m_config.initialCreaturesCount; ++i)
    {
      b2Vec2 position(distX(m_generator), distY(m_generator));
      auto nn = Creature::makeNeuralNetork(creatureNeuralNetworkConfig, m_generator);
      m_creatures.emplace_back(m_world, m_creatureConfig, std::move(nn), position, m_creatureConfig.maxEnergy, m_creatureConfig.maxHealth);
    }
  }

  // Berry Bushes
  {
    std::uniform_real_distribution<double> distX(-m_config.width/2.0f,  m_config.width/2.0f);
    std::uniform_real_distribution<double> distY(-m_config.height/2.0f, m_config.height/2.0f);
    std::uniform_int_distribution<size_t> distSize(m_config.initialBerryBushesClusterSizeMin, m_config.initialBerryBushesClusterSizeMax);

    for(size_t i=0; i<m_config.initialBerryBushesClusterCount; ++i)
    {
      size_t clusterSize = distSize(m_generator);
      for(size_t j=0; j<clusterSize; ++j)
      {
        b2Vec2 position(distX(m_generator), distY(m_generator));
        m_berryBushes.emplace_back(m_world, m_berryBushConfig, position);
      }
    }
  }
}

void World::update(float dt)
{
  m_remaingUpdateTime += dt;

  // While it is okay to do physics at 60FPS
  // Doing creature update at 60FPS is slow
  // TODO: Fixme
  const float timeStep = 1.0f / 60.0f;
  while(m_remaingUpdateTime >= timeStep)
  {
    m_remaingUpdateTime -= timeStep;

    m_updateTimer.begin();

    // 1: Physics
    const int32 velocityIterations = 8;
    const int32 positionIterations = 3;
    m_world.Step(timeStep, velocityIterations, positionIterations);

    // 2: Berry Bushes and Creatures
    for(auto& berryBush : m_berryBushes)
      berryBush.update(timeStep);

    for(auto& creature : m_creatures)
      creature.updatePerception(timeStep);

#pragma omp parallel for
    for(size_t i=0; i<m_creatures.size(); ++i)
      m_creatures[i].updateNeuralNetwork();

    for(auto& creature : m_creatures)
      creature.update(timeStep, *this);

    // 3: Clear dead creatures
    auto oldSize = m_creatures.size();
    m_creatures.erase(std::remove_if(m_creatures.begin(), m_creatures.end(), std::mem_fn(&Creature::dead)), m_creatures.end());
    auto newSize = m_creatures.size();
    m_info.deathToll += oldSize - newSize;

    // 4: Add new creatures
    m_info.birthCount += m_newborns.size();
    m_creatures.insert(m_creatures.end(),
      std::move_iterator(m_newborns.begin()),
      std::move_iterator(m_newborns.end())
    );
    m_newborns.clear();

    // 5: Update info
    {
      m_info.ageStatistics = Statistics<float, float>(m_creatures.begin(), m_creatures.end(), [](const Creature& creature){
        return creature.statistics().lifetime;
      });

      m_info.matingCountStatistics = Statistics<float, size_t>(m_creatures.begin(), m_creatures.end(), [](const Creature& creature){
        return creature.statistics().matingCount;
      });

      m_info.healthyCreaturesCount =  std::count_if(m_creatures.begin(), m_creatures.end(), std::mem_fn(&Creature::healthy));
      m_info.creaturesCount = m_creatures.size();

      m_info.realTime = m_startTime.getElapsedTime().asSeconds();
      m_info.worldTime += timeStep;

      m_info.averageUpdateTime = m_updateTimer.average();
    }

    m_updateTimer.end();
  }
}

namespace
{
  template<typename... Args>
  std::string concatenate(const Args&... args)
  {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);
    (ss << ... << args);
    return ss.str();
  }
}

void World::draw(Renderer& renderer) const
{
  renderer.addGuiText(concatenate("Creatures:", m_info.healthyCreaturesCount, "/", m_info.creaturesCount, "(Healthy/All)"));
  renderer.addGuiText(concatenate("Statistics:", m_info.deathToll, "/", m_info.birthCount, "(DeathToll/BirthCount)"));

  renderer.addGuiText(concatenate("Age:", m_info.ageStatistics.toString()));
  renderer.addGuiText(concatenate("Mating Count:", m_info.matingCountStatistics.toString()));

  renderer.addGuiText(concatenate("Time:", m_info.realTime, "/", m_info.worldTime, "/", m_info.worldTime / m_info.realTime, "(Real/World/Ratio)"));
  renderer.addGuiText(concatenate("Update Time:", m_info.averageUpdateTime));

  renderer.addRectangle(sf::Vector2f(0.0f, 0.0f), {m_config.width, m_config.height}, sf::Color::White);

  for(const auto& berryBush: m_berryBushes)
    berryBush.draw(renderer);

  for(const auto& creature: m_creatures)
    creature.draw(renderer);
}

void World::beginStatistics(CSVFile& file) const
{
  file.write(
    "Time",
    "Creatures Count - Healthy",
    "Creatures Count - All",
    "Age - Mean", "Age - StandardDeviation",
    "Age - Minimum", "Age - LowerQuartile",
    "Age - Median",
    "Age - UpperQuartile", "Age - Maximum" ,
    "Mating Count - Mean", "Mating Count - StandardDeviation",
    "Mating Count - Minimum", "Mating Count - LowerQuartile",
    "Mating Count - Median",
    "Mating Count - UpperQuartile", "Mating Count - Maximum"
  );
}

void World::writeStatistics(CSVFile& file) const
{
  file.write(
    m_info.worldTime,

    m_info.healthyCreaturesCount,
    m_info.creaturesCount,

    m_info.ageStatistics.mean, m_info.ageStatistics.standardDeviation,
    m_info.ageStatistics.minimum, m_info.ageStatistics.lowerQuartile,
    m_info.ageStatistics.median,
    m_info.ageStatistics.upperQuartile, m_info.ageStatistics.maximum ,

    m_info.matingCountStatistics.mean, m_info.matingCountStatistics.standardDeviation,
    m_info.matingCountStatistics.minimum, m_info.matingCountStatistics.lowerQuartile,
    m_info.matingCountStatistics.median,
    m_info.matingCountStatistics.upperQuartile, m_info.matingCountStatistics.maximum
  );
}

