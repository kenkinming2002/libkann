#include "World.hpp"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RectangleShape.hpp>

#include <iostream>
#include <algorithm>

#include "Config.hpp"

World::World(seed_type seed) : m_dimension(CONFIG.world.width, CONFIG.world.height), m_generator(seed)
{
  std::uniform_int_distribution<typename Creature::random_engine_type::result_type> seedDistribution(
    Creature::random_engine_type::min(), 
    Creature::random_engine_type::max()
  );

  std::uniform_real_distribution<double> xPositionDistribution(-m_dimension(0)/2.0, m_dimension(0)/2.0);
  std::uniform_real_distribution<double> yPositionDistribution(-m_dimension(1)/2.0, m_dimension(1)/2.0);

  for(size_t i=0; i<CONFIG.world.initialCreaturesCount; ++i)
    m_creatures.emplace_back(seedDistribution(m_generator), 
      Eigen::Vector2d(
        xPositionDistribution(m_generator), 
        yPositionDistribution(m_generator)
      )
    );

  for(size_t i=0; i<CONFIG.world.initialBerryBushesCount; ++i)
    m_berryBushes.emplace_back(Eigen::Vector2d(
        xPositionDistribution(m_generator), 
        yPositionDistribution(m_generator)
      )
    );
}

void World::update(float dt)
{
  std::uniform_int_distribution<typename Creature::random_engine_type::result_type> seedDistribution(
    Creature::random_engine_type::min(), 
    Creature::random_engine_type::max()
  );

#pragma omp parallel for
  for(auto& creature: m_creatures)
    creature.updateSight(*this);

#pragma omp parallel for
  for(auto& creature: m_creatures)
    creature.update(dt, m_dimension);

  m_creatures.erase(std::remove_if(m_creatures.begin(), m_creatures.end(), [](const auto& creature){ return creature.dead(); }), m_creatures.end());

#pragma omp parallel for
  for(auto& berryBush: m_berryBushes)
    berryBush.update(dt);

  for(auto& creature: m_creatures)
    for(auto& berryBush: m_berryBushes)
      if(creature.canEat(berryBush))
        creature.eat(berryBush);

  std::vector<Creature> newborns;
  for(size_t i=0; i<m_creatures.size(); ++i)
    for(size_t j=0; j<m_creatures.size(); ++j)
      if(i!=j && Creature::canMate(m_creatures[i], m_creatures[j]))
        if(auto newborn = Creature::mate(m_creatures[i], m_creatures[j], seedDistribution(m_generator)))
          newborns.push_back(*newborn);

  m_creatures.insert(m_creatures.end(), newborns.begin(), newborns.end());
}

void World::draw(sf::RenderTarget &target, sf::RenderStates states) const
{
  for(const auto& berryBush: m_berryBushes)
    target.draw(berryBush, states);

  for(const auto& creature: m_creatures)
    target.draw(creature, states);

  sf::RectangleShape rectangleShape;

  rectangleShape.setPosition(0.0f, 0.0f);
  rectangleShape.setSize({static_cast<float>(m_dimension(0)), static_cast<float>(m_dimension(1))});
  rectangleShape.setOrigin(m_dimension(0)/2.0, m_dimension(1)/2.0);

  rectangleShape.setOutlineThickness(5.0f);
  rectangleShape.setOutlineColor(sf::Color::Black);
  rectangleShape.setFillColor(sf::Color::Transparent);

  target.draw(rectangleShape, states);
}

void World::log() const
{
  std::cout << "Creature count:" << m_creatures.size() << '\n';
  std::cout << "Healthy Creature count:" << std::count_if(m_creatures.begin(), m_creatures.end(), [](const auto& creature){
      return creature.health() == CONFIG.creature.maxHealth;
  }) << '\n';
}
