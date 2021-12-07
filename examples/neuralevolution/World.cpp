#include "World.hpp"
#include "libkann/NeuralNetwork.hpp"
#include "libkann/RecurrentNeuralNetwork.hpp"

#include <SFML/Graphics/RenderTarget.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>

static constexpr double AVERAGE_COUNT_PER_CELL = 3.0;

World::World(Config config, Creature::Config creatureConfig, Creature::NeuralNetworkConfig creatureNeuralNetworkConfig, BerryBush::Config berryBushConfig)
  : m_config(config), m_creatureConfig(creatureConfig), m_berryBushConfig(berryBushConfig),
    m_generator(config.seed),
    m_world(b2Vec2(0.0f,0.0f)) // No gravity
{
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

    const int32 velocityIterations = 8;
    const int32 positionIterations = 3;
    m_world.Step(timeStep, velocityIterations, positionIterations);

    m_worldTime += timeStep;

    for(auto& berryBush : m_berryBushes)
      berryBush.update(timeStep);

    for(auto& creature : m_creatures)
      creature.updatePerception(timeStep);

#pragma omp parallel for
    for(size_t i=0; i<m_creatures.size(); ++i)
      m_creatures[i].updateNeuralNetwork();

    for(auto& creature : m_creatures)
      creature.update(timeStep, *this);

    auto oldSize = m_creatures.size();
    m_creatures.erase(std::remove_if(m_creatures.begin(), m_creatures.end(), std::mem_fn(&Creature::dead)), m_creatures.end());
    auto newSize = m_creatures.size();
    m_deathToll += oldSize - newSize;

    m_updateTimer.end();
  }
}

World::Info World::info() const
{
  Info info;

  info.ageStatistics = Statistics<float, float>(m_creatures.begin(), m_creatures.end(), [](const Creature& creature){
      return creature.statistics().lifetime;
  });

  info.matingCountStatistics = Statistics<float, size_t>(m_creatures.begin(), m_creatures.end(), [](const Creature& creature){
      return creature.statistics().matingCount;
  });

  info.healthyCreaturesCount =  std::count_if(m_creatures.begin(), m_creatures.end(), std::mem_fn(&Creature::healthy));
  info.creaturesCount = m_creatures.size();

  info.deathToll = m_deathToll;
  info.birthCount = m_birthCount;

  info.realTime = m_startTime.getElapsedTime().asSeconds();
  info.worldTime = m_worldTime;

  info.averageUpdateTime = m_updateTimer.average();

  return info;
}

