#include "Creature.hpp"

#include "World.hpp"
#include "Config.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderTarget.hpp>

#include <iostream>
#include <cmath>

static sf::Color lerp(sf::Color a, sf::Color b, float t)
{
  return sf::Color(
    std::lerp(a.r, b.r, t),
    std::lerp(a.g, b.g, t),
    std::lerp(a.b, b.b, t)
  );
}

Creature::Creature(typename NeuralNetwork::seed_type seed, Eigen::Vector2d position) 
  : m_neuralNetwork({NUM_INPUT, 50, 50, 50, 50, 50, 50, NUM_OUTPUT}, seed), 
    m_position(position), m_energy(CONFIG.creature.maxEnergy), m_health(CONFIG.creature.maxHealth) {}

Creature::Creature(NeuralNetwork neuralNetwork, Eigen::Vector2d position, double energy) 
  : m_neuralNetwork(std::move(neuralNetwork)),
    m_position(position), m_energy(energy), m_health(CONFIG.creature.maxHealth) {}

void Creature::updateSight(const World& world)
{
  /// 1: Closest Creature
  auto distance_to = [this](const Creature& rhs) {
    static_assert(std::numeric_limits<double>::is_iec559);
    if(this == &rhs)
      return std::numeric_limits<double>::infinity();
    else
      return Creature::distance(*this, rhs);
  };

  const Creature& closestCreature = *std::min_element(world.creatures().begin(), world.creatures().end(), [&](const auto& lhs, const auto& rhs){ 
    return distance_to(lhs) < distance_to(rhs);
  });
  m_sight.closestCreatureLocation = closestCreature.position();

  /// 2: Closest Berry Bush
  const BerryBush& closestBerryBush = *std::min_element(world.berryBushes().begin(), world.berryBushes().end(), [this](const auto& lhs, const auto& rhs){
      return this->distance(lhs) < this->distance(rhs);
  });

  if(m_sight.closestBerryBush)
    --(*m_sight.closestBerryBush->TARGETED_COUNT);

  m_sight.closestBerryBush = &closestBerryBush;
  ++(*m_sight.closestBerryBush->TARGETED_COUNT);

  m_sight.closestBerryBushLocation = closestBerryBush.position();
}

void Creature::update(float dt, Eigen::Vector2d worldDimension)
{
  // 1: Neural network
  // Prepare input
  Eigen::Vector2d closesetBerryBushOffset = Eigen::Rotation2Dd(-m_rotation) * (m_sight.closestBerryBushLocation - m_position);
  Eigen::Vector2d closesetCreatureOffset  = Eigen::Rotation2Dd(-m_rotation) * (m_sight.closestCreatureLocation - m_position);

  m_neuralNetwork.input({m_energy, m_health, closesetBerryBushOffset(0), closesetBerryBushOffset(1), closesetCreatureOffset(0), closesetCreatureOffset(1)});

  // Feed Forwrad
  m_neuralNetwork.feedForward();

  // Parse output
  auto output = m_neuralNetwork.output();

  double linearSpeed = output[0] >= 0.0 ? output[0] * CONFIG.creature.forwardLinearSpeed : output[0] * CONFIG.creature.backwardLinearSpeed;
  double angularSpeed = output[1] * CONFIG.creature.angularSpeed;
  m_eatingDesire = output[2];
  m_matingDesire = output[3];

  // 2: Movement
  m_rotation += angularSpeed;
  m_position += (Eigen::Rotation2Dd(m_rotation) * Eigen::Vector2d(linearSpeed, 0.0)) * dt;

  m_position(0) = std::clamp(m_position(0), -worldDimension(0)/2.0, worldDimension(0)/2.0);
  m_position(1) = std::clamp(m_position(1), -worldDimension(1)/2.0, worldDimension(1)/2.0);

  // 3: Energy, health and suvival
  double energyDrain = (CONFIG.creature.passiveEnergyDrain + CONFIG.creature.movementEnergyDrainMultiplier * linearSpeed * linearSpeed) * dt;

  if(!takeEnergy(energyDrain))
    takeHealth(CONFIG.creature.hungerHealthDrain * dt);

  m_eatingCooldown -= dt;
  if(m_eatingCooldown<0.0f)
    m_eatingCooldown=0.0f;

  m_matingCooldown -= dt;
  if(m_matingCooldown<0.0f)
    m_matingCooldown=0.0f;
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

bool Creature::canEat(const BerryBush& berryBush) const
{
  static constexpr double EATING_DISTANCE = 50.0;

  return m_eatingDesire > 0.0 &&
         CONFIG.creature.maxEnergy - m_energy >= CONFIG.berryBush.energyPerBerry &&
         (m_position - berryBush.position()).norm() < EATING_DISTANCE &&
         berryBush.count() != 0 &&
         m_eatingCooldown == 0.0f;
}

void Creature::eat(BerryBush& berryBush)
{
  m_eatingCooldown = CONFIG.creature.eatingCooldown;

  berryBush.take();
  m_energy += CONFIG.berryBush.energyPerBerry;
}

bool Creature::canMate(const Creature& lhs, const Creature& rhs)
{
  static constexpr double MATING_DISTANCE = 20.0;
  return lhs.m_matingDesire > 0.0 && 
         rhs.m_matingDesire > 0.0 &&
         (lhs.m_position - rhs.m_position).norm() < MATING_DISTANCE &&
         lhs.m_matingCooldown == 0.0f && rhs.m_matingCooldown == 0.0f;
}

std::optional<Creature> Creature::mate(Creature& lhs, Creature& rhs, seed_type seed)
{
  lhs.m_matingCooldown = CONFIG.creature.matingCooldown;
  rhs.m_matingCooldown = CONFIG.creature.matingCooldown;

  if(!lhs.takeEnergy(CONFIG.creature.maxEnergy * 0.2) || !rhs.takeEnergy(CONFIG.creature.maxEnergy * 0.2))
    return std::nullopt; // Mating without sufficient energy does nothing

  //TODO: implement inheritance
  return Creature(NeuralNetwork::cross(lhs.m_neuralNetwork, rhs.m_neuralNetwork, seed, CONFIG.neuralNetwork.mutationRate), (lhs.m_position+rhs.m_position)/2.0, 0.3 * CONFIG.creature.maxEnergy);
}

void Creature::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
  sf::CircleShape circleShape;
  {
    static constexpr float RADIUS_MULTIPIER = 10.0f;
    float radius = (m_health / CONFIG.creature.maxHealth) * RADIUS_MULTIPIER;

    circleShape.setRadius(radius);
    circleShape.setOrigin({radius, radius});

    circleShape.setFillColor(lerp(sf::Color::Yellow, sf::Color::Green, m_energy / CONFIG.creature.maxEnergy));

    circleShape.setOutlineThickness(2);
    circleShape.setOutlineColor(sf::Color::Black);
  }
  circleShape.setPosition(m_position(0), m_position(1));
  target.draw(circleShape, states);

  sf::RectangleShape rectangleShape;
  {
    static constexpr float THICKNESS = 3.0f;
    static constexpr float LENGTH = 30.0f;

    rectangleShape.setSize({LENGTH, THICKNESS});
    rectangleShape.setOrigin(0.0f, THICKNESS/2.0f);

    rectangleShape.setFillColor(sf::Color::Red);
    rectangleShape.setRotation(360.0f * m_rotation / (2*M_PI));
  }
  rectangleShape.setPosition(m_position(0), m_position(1));
  target.draw(rectangleShape, states);
}

std::ostream& operator<<(std::ostream& os, const Creature& creature)
{
  os << "Position:" << creature.m_position(0) << ", " << creature.m_position(1) << '\n';
  os << "Energy:" << creature.m_energy << '\n';
  os << "Health:" << creature.m_health;
  return os;
}
