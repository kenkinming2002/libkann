#pragma once

#include "Creature.hpp"
#include "BerryBush.hpp"
#include "Grid.hpp"

#include <vector>
#include <random>

#include <SFML/System/Clock.hpp>

class World
{
public:
  using random_engine_type = std::mt19937;
  using seed_type = random_engine_type::result_type;

public:
  World(seed_type seed);

public:
  friend class Renderer;

public:
  void update(float dt);

public:
  struct Info
  {
    size_t healthyCreaturesCount;
    size_t creaturesCount;

    size_t deathToll;
    size_t birthCount;

    float realTime;
    float worldTime;
  };
  Info info() const;

public:
  const auto& creatures() const { return m_creatures; }
  auto& creatures() { return m_creatures; }
  const auto& berryBushes() const { return m_berryBushes; }
  auto& berryBushes() { return m_berryBushes; }

public:
  auto dimension() const { return m_dimension; }

public:
  void addCreature(const Creature& creature) { m_newborns.push_back(creature); }

public:
  auto& prng() { return m_generator; }

private:
  Eigen::Vector2d m_dimension;
  Grid<Creature> m_creatures;
  Grid<BerryBush> m_berryBushes;

private:
  std::vector<Creature> m_newborns;

private:
  std::mt19937 m_generator;

private:
  size_t m_deathToll = 0;
  size_t m_birthCount = 0;

private:
  sf::Clock m_startTime;
  float m_worldTime = 0.0f;
};
