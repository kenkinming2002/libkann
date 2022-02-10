#pragma once

#include "Creature.hpp"
#include "BerryBush.hpp"
#include "Timer.hpp"
#include "Statistics.hpp"
#include "CSVFile.hpp"

#include <vector>
#include <random>
#include <variant>

#include <SFML/System/Clock.hpp>

#include <box2d/box2d.h>

class World
{
public:
  using random_engine_type = std::default_random_engine;
  using seed_type = random_engine_type::result_type;

public:
  struct Config
  {
    seed_type seed;

    float width;
    float height;

    size_t initialCreaturesCount;
    size_t initialBerryBushesClusterSizeMin;
    size_t initialBerryBushesClusterSizeMax;
    size_t initialBerryBushesClusterCount;
  };
  World(Config config, Creature::Config creatureConfig,
      BerryBush::Config berryBushConfig);

public:
  friend class Renderer;

public:
  void update(float dt);

public:
  void draw(Renderer& renderer) const;

public:
  struct Info
  {
    Statistics<float, float> ageStatistics;
    Statistics<float, size_t> matingCountStatistics;

    size_t healthyCreaturesCount;
    size_t creaturesCount;

    size_t deathToll;
    size_t birthCount;

    float realTime;
    float worldTime;

    float averageUpdateTime;
  };

public:
  void beginStatistics(CSVFile& file) const;
  void writeStatistics(CSVFile& file) const;

public:
  b2Vec2 dimension() const { return {m_config.width, m_config.height}; }

public:
  void addCreature(Creature creature) { m_newborns.push_back(std::move(creature)); }

public:
  auto& prng() { return m_generator; }
  auto& world() { return m_world; }

private:
  const Config m_config;
  const Creature::Config m_creatureConfig;
  const BerryBush::Config m_berryBushConfig;

private:
  std::default_random_engine m_generator;

private:
  b2World m_world;

private:
  std::vector<BerryBush> m_berryBushes;
  std::vector<Creature> m_creatures;
  std::vector<Creature> m_newborns;

private:
  Info m_info{
    .deathToll = 0,
    .birthCount = 0,
    .worldTime = 0.0f
  };

private:
  sf::Clock m_startTime;

private:
  Timer<> m_updateTimer;
};
