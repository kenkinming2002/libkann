#include "Creature.hpp"

#include "World.hpp"
#include "Config.hpp"
#include "Ray.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderTarget.hpp>

#include <iostream>
#include <cmath>
#include <limits>

static sf::Color lerp(sf::Color a, sf::Color b, float t)
{
  return sf::Color(
    std::lerp(a.r, b.r, t),
    std::lerp(a.g, b.g, t),
    std::lerp(a.b, b.b, t)
  );
}

template<typename PRNG>
Creature::Creature(PRNG& prng, Eigen::Vector2d position) 
  : m_neuralNetwork({NUM_INPUT, 50, 50, 50, 50, 50, 50, NUM_OUTPUT}, prng), 
    m_position(position), m_energy(CONFIG.creature.maxEnergy), m_health(CONFIG.creature.maxHealth) {}

template Creature::Creature(std::mt19937& prng, Eigen::Vector2d position);

Creature::Creature(NeuralNetwork neuralNetwork, Eigen::Vector2d position, double energy) 
  : m_neuralNetwork(neuralNetwork),
    m_position(position), m_energy(energy), m_health(CONFIG.creature.maxHealth) {}

void Creature::updateSight(World& world) const
{
  for(Eye& eye: m_eyes)
  {
    eye.distance = eye.viewDistance;
    Ray ray{m_position, m_rotation + eye.angle};

    for(auto& creature: world.creatures())
    {
      if(&creature == this)
        continue;

      CircleCollider circleCollider{creature.m_position, CONFIG.creature.radius};
      if(double distance = ray.cast(circleCollider); distance < eye.distance && distance > 0.0)
      {
        eye.distance = distance;
        eye.target = std::ref(creature);
      }
    }

    for(auto& berryBush: world.berryBushes())
    {
      CircleCollider circleCollider{berryBush.position(), CONFIG.berryBush.radius};
      if(double distance = ray.cast(circleCollider); distance < eye.distance && distance > 0.0)
      {
        eye.distance = distance;
        eye.target = std::ref(berryBush);
      }
    }
  }
}

void Creature::preUpdate(float /*dt*/, World& world) const
{
  updateSight(world);

  // 1: Neural network
  // Prepare input
  m_neuralNetwork.input({m_energy, m_health, m_eyes[0].distance, m_eyes[1].distance});
  //m_neuralNetwork.input({m_energy, m_health, 0.0, 0.0});

  // Feed Forwrad
  m_neuralNetwork.feedForward();
}

void Creature::update(float dt, World& world)
{
  // Parse output
  auto output = m_neuralNetwork.output();

  double linearSpeed = output[0] >= 0.0 ? output[0] * CONFIG.creature.forwardLinearSpeed : output[0] * CONFIG.creature.backwardLinearSpeed;
  double angularSpeed = output[1] * CONFIG.creature.angularSpeed;
  m_eatingDesire = output[2];
  m_matingDesire = output[3];

  // 2: Movement
  m_rotation += angularSpeed;
  m_position += (Eigen::Rotation2Dd(m_rotation) * Eigen::Vector2d(linearSpeed, 0.0)) * dt;

  m_position(0) = std::clamp(m_position(0), -world.dimension()(0)/2.0, world.dimension()(0)/2.0);
  m_position(1) = std::clamp(m_position(1), -world.dimension()(1)/2.0, world.dimension()(1)/2.0);

  // 3: Energy, health and suvival
  double energyDrain = (CONFIG.creature.passiveEnergyDrain + CONFIG.creature.movementEnergyDrainMultiplier * linearSpeed * linearSpeed) * dt;

  if(!takeEnergy(energyDrain))
    takeHealth(CONFIG.creature.hungerHealthDrain * dt);

  if(m_energy >= CONFIG.creature.maxEnergy * CONFIG.creature.healingThreshold)
  {
    double amount = std::min(CONFIG.creature.maxHealth - m_health, m_energy);
    m_health += amount;
    m_energy -= amount;
  }

  m_eatingCooldown -= dt;
  if(m_eatingCooldown<0.0f)
    m_eatingCooldown=0.0f;

  m_matingCooldown -= dt;
  if(m_matingCooldown<0.0f)
    m_matingCooldown=0.0f;

  updateEating();
  updateMating(world);
}

void Creature::updateEating()
{
  static constexpr double EATING_ENERGY_COST = 5.0f;
  const double EATING_DISTANCE = CONFIG.creature.radius * 1.5 + CONFIG.berryBush.radius;

  if(m_eatingDesire<0.0)
    return; // Don't want to eat

  if(m_eatingCooldown!=0.0)
    return; // Cooldown
  
  this->takeEnergy(EATING_ENERGY_COST);
  m_eatingCooldown = CONFIG.creature.eatingCooldown;

  for(auto& eye: m_eyes)
  {
    if(!std::holds_alternative<std::reference_wrapper<BerryBush>>(eye.target))
      continue;

    auto& berryBush = std::get<std::reference_wrapper<BerryBush>>(eye.target).get();

    if(auto distance = (m_position - berryBush.position()).norm();
        distance >= EATING_DISTANCE)
      continue;

    if(berryBush.count() == 0)
      continue; // No berry to eat

    berryBush.take();
    m_energy += CONFIG.berryBush.energyPerBerry;
    if(m_energy>CONFIG.creature.maxEnergy)
    {
      this->takeHealth(m_energy-CONFIG.creature.maxEnergy);
      m_energy = CONFIG.creature.maxEnergy;
    }

    return;
  }
}

void Creature::updateMating(World& world)
{
  static constexpr double MATING_ENERGY_COST = 5.0f;
  const double MATING_DISTANCE = CONFIG.creature.radius * 2.5;

  for(auto& eye: m_eyes)
  {
    if(!std::holds_alternative<std::reference_wrapper<Creature>>(eye.target))
      continue;

    auto& otherCreature = std::get<std::reference_wrapper<Creature>>(eye.target).get();

    if(m_matingDesire<0.0 || otherCreature.m_matingDesire<0.0)
      return; // Don't want to mate

    if(m_matingCooldown!=0.0 || otherCreature.m_matingCooldown != 0.0)
      return; // Cooldown

    this->takeEnergy(MATING_ENERGY_COST);

    m_matingCooldown = CONFIG.creature.matingCooldown;
    otherCreature.m_matingCooldown = CONFIG.creature.matingCooldown;

    if(auto distance = (m_position - otherCreature.m_position).norm();
        distance >= MATING_DISTANCE)
      continue; // Too far away

    if(!takeEnergy(CONFIG.creature.maxEnergy * 0.2) || !otherCreature.takeEnergy(CONFIG.creature.maxEnergy * 0.2))
      continue;

    auto neuralNetwork = NeuralNetwork::cross(m_neuralNetwork, otherCreature.m_neuralNetwork, world.prng(), CONFIG.neuralNetwork.mutationRate);
    Eigen::Vector2d position = (m_position + otherCreature.m_position) / 2.0;

    world.addCreature(Creature(std::move(neuralNetwork), position, 0.3 * CONFIG.creature.maxEnergy));

    return;
  }
}

bool Creature::takeEnergy(double amount)
{
  m_energy -= amount;
  if(m_energy<0.0)
    m_energy=0.0;

  return m_energy != 0.0;
}

bool Creature::takeHealth(double amount)
{
  m_health -= amount;
  if(m_health<0.0)
    m_health=0.0;

  return m_health != 0.0;
}

void Creature::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
  sf::CircleShape circleShape;
  {
    float radius = CONFIG.creature.radius * m_health / CONFIG.creature.maxHealth;

    circleShape.setRadius(radius);
    circleShape.setOrigin({radius, radius});

    circleShape.setFillColor(lerp(sf::Color::Yellow, sf::Color::Green, m_energy / CONFIG.creature.maxEnergy));

    circleShape.setOutlineThickness(2);
    circleShape.setOutlineColor(sf::Color::Black);
  }
  circleShape.setPosition(m_position(0), m_position(1));
  target.draw(circleShape, states);

  static constexpr float THICKNESS = 3.0f;

  for(const auto& eye: m_eyes)
  {
    sf::RectangleShape rectangleShape;
    {
      rectangleShape.setSize({static_cast<float>(eye.distance), THICKNESS});
      rectangleShape.setOrigin(0.0f, THICKNESS/2.0f);

      rectangleShape.setFillColor(sf::Color::Red);
      rectangleShape.setRotation((m_rotation+eye.angle) * 360.0 / (2*M_PI));
    }
    rectangleShape.setPosition(m_position(0), m_position(1));
    target.draw(rectangleShape, states);
  }
}

std::ostream& operator<<(std::ostream& os, const Creature& creature)
{
  os << "Position:" << creature.m_position(0) << ", " << creature.m_position(1) << '\n';
  os << "Energy:" << creature.m_energy << '\n';
  os << "Health:" << creature.m_health;
  return os;
}
