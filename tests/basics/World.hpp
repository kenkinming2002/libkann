#pragma once

#include "Creature.hpp"
#include "BerryBush.hpp"

#include <vector>
#include <random>

#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/View.hpp>
#include <SFML/Window/Event.hpp>

class World : public sf::Drawable
{
public:
  using random_engine_type = std::mt19937;
  using seed_type = random_engine_type::result_type;

public:
  static constexpr float ZOOM_SPEED = 2.0f;
  static constexpr float MOVE_SPEED = 10.0f;

public:
  World(sf::View view, seed_type seed);

public:
  bool handleInput(sf::Event event);
  void update(float dt);
  void draw(sf::RenderTarget &target, sf::RenderStates states) const override;

public:
  void log() const;

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
  const sf::View m_defaultView;
  sf::View m_view;
  float m_scale = 1.0f;

private:
  Eigen::Vector2d m_dimension;
  std::vector<Creature> m_creatures;
  std::vector<BerryBush> m_berryBushes;

private:
  std::vector<Creature> m_newborns;

private:
  std::mt19937 m_generator;
};
