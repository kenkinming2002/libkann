#pragma once

#include "Creature.hpp"
#include "BerryBush.hpp"

#include <vector>
#include <random>
#include <SFML/Graphics/Drawable.hpp>

class World : public sf::Drawable
{
public:
  using random_engine_type = std::mt19937;
  using seed_type = typename random_engine_type::result_type;

public:
  World(size_t creaturesCount, size_t berryBushesCount, Eigen::Vector2d dimension, seed_type seed);

public:
  void update(float dt);
  void draw(sf::RenderTarget &target, sf::RenderStates states) const override;

public:
  void log() const;

public:
  const auto& creatures() const { return m_creatures; }
  const auto& berryBushes() const { return m_berryBushes; }

private:
  Eigen::Vector2d m_dimension;
  std::vector<Creature> m_creatures;
  std::vector<BerryBush> m_berryBushes;

private:
  std::mt19937 m_generator;

};
