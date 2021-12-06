#pragma once

#include "Creature.hpp"
#include "BerryBush.hpp"
#include "Grid.hpp"
#include "Timer.hpp"
#include "Statistics.hpp"

#include <vector>
#include <random>
#include <variant>

#include <SFML/System/Clock.hpp>

class World
{
public:
  using random_engine_type = std::default_random_engine;
  using seed_type = random_engine_type::result_type;

public:
  struct Config
  {
    seed_type seed;

    double width;
    double height;

    size_t initialCreaturesCount;
    size_t initialBerryBushesClusterSizeMin;
    size_t initialBerryBushesClusterSizeMax;
    size_t initialBerryBushesClusterCount;
  };
  World(Config config, Creature::Config creatureConfig,
      Creature::NeuralNetworkConfig creatureNeuralNetworkConfig,
      BerryBush::Config berryBushConfig);

public:
  friend class Renderer;

public:
  void update(float dt);

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
  Info info() const;

public:
  using result_variant = std::variant<std::monostate, std::reference_wrapper<Creature>, std::reference_wrapper<BerryBush>>;
  result_variant find(Eigen::Vector2d position);

public:
  const auto& creatures() const { return m_creatures; }
  auto& creatures() { return m_creatures; }
  const auto& berryBushes() const { return m_berryBushes; }
  auto& berryBushes() { return m_berryBushes; }

public:
  Eigen::Vector2d dimension() const { return {m_config.width, m_config.height}; }

public:
  void addCreature(Creature creature) { m_newborns.push_back(std::move(creature)); }

public:
  auto& prng() { return m_generator; }

private:
  const Config m_config;
  const Creature::Config m_creatureConfig;
  const BerryBush::Config m_berryBushConfig;

private:
  Grid<Creature> m_creatures;
  Grid<BerryBush> m_berryBushes;

private:
  std::vector<Creature> m_newborns;

private:
  std::default_random_engine m_generator;

private:
  size_t m_deathToll = 0;
  size_t m_birthCount = 0;

private:
  sf::Clock m_startTime;
  float m_worldTime = 0.0f;

private:
  Timer<> m_updateTimer;
};
