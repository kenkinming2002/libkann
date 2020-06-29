#include "Creature.hpp"

#include "World.hpp"

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
    m_position(position), m_energy(MAX_ENERGY), m_health(MAX_HEALTH) {}

Creature::Creature(NeuralNetwork neuralNetwork, Eigen::Vector2d position, double energy) 
  : m_neuralNetwork(std::move(neuralNetwork)),
    m_position(position), m_energy(energy), m_health(MAX_HEALTH) {}

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
  static constexpr double FORWARD_LINEAR_SPEED_MULTIPLIER = 100.0;
  static constexpr double BACKWARD_LINEAR_SPEED_MULTIPLIER = 80.0;
  static constexpr double ANGULAR_SPEED_MULTIPLIER = 0.1;

  static constexpr double PASSIVE_ENERGY_DRAIN = 2.0;
  static constexpr double MOVEMENT_ENERGY_DRAIN_MULTIPIER = 0.001;

  static constexpr double HUNGER_HEALTH_DRAIN = 5.0;

  // 1: Neural network
  // Prepare input
  Eigen::Vector2d closesetBerryBushOffset = Eigen::Rotation2Dd(-m_rotation) * (m_sight.closestBerryBushLocation - m_position);
  Eigen::Vector2d closesetCreatureOffset  = Eigen::Rotation2Dd(-m_rotation) * (m_sight.closestCreatureLocation - m_position);

  m_neuralNetwork.input({m_energy, m_health, closesetBerryBushOffset(0), closesetBerryBushOffset(1), closesetCreatureOffset(0), closesetCreatureOffset(1)});

  // Feed Forwrad
  m_neuralNetwork.feedForward();

  // Parse output
  auto output = m_neuralNetwork.output();
  double linearSpeed = output[0] >= 0.0 ? output[0] * FORWARD_LINEAR_SPEED_MULTIPLIER : output[0] * BACKWARD_LINEAR_SPEED_MULTIPLIER;
  double angularSpeed = output[1] * ANGULAR_SPEED_MULTIPLIER;
  m_eatingDesire = output[2];
  m_matingDesire = output[3];

  // 2: Movement
  m_rotation += angularSpeed;
  m_position += (Eigen::Rotation2Dd(m_rotation) * Eigen::Vector2d(linearSpeed, 0.0)) * dt;

  m_position(0) = std::clamp(m_position(0), -worldDimension(0)/2.0, worldDimension(0)/2.0);
  m_position(1) = std::clamp(m_position(1), -worldDimension(1)/2.0, worldDimension(1)/2.0);

  // 3: Energy, health and suvival
  double energyDrain = (PASSIVE_ENERGY_DRAIN + MOVEMENT_ENERGY_DRAIN_MULTIPIER * linearSpeed * linearSpeed) * dt;

  if(!takeEnergy(energyDrain))
    takeHealth(HUNGER_HEALTH_DRAIN * dt);

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
  static constexpr double EATING_DESIRE_THRESHOLD = 0.0;

  return m_eatingDesire > EATING_DESIRE_THRESHOLD &&
         MAX_ENERGY - m_energy >= BerryBush::ENERGY_PER_BERRY &&
         (m_position - berryBush.position()).norm() < EATING_DISTANCE &&
         berryBush.count() != 0 &&
         m_eatingCooldown == 0.0f;
}

void Creature::eat(BerryBush& berryBush)
{
  m_eatingCooldown = EATING_COOLDOWN;

  berryBush.take();
  m_energy += BerryBush::ENERGY_PER_BERRY;
}

bool Creature::canMate(const Creature& lhs, const Creature& rhs)
{
  static constexpr double MATING_DISTANCE = 20.0;
  static constexpr double MATING_DESIRE_THRESHOLD = -0.25;
  return lhs.m_matingDesire > MATING_DESIRE_THRESHOLD && 
         rhs.m_matingDesire > MATING_DESIRE_THRESHOLD &&
         (lhs.m_position - rhs.m_position).norm() < MATING_DISTANCE &&
         lhs.m_matingCooldown == 0.0f && rhs.m_matingCooldown == 0.0f;
}

std::optional<Creature> Creature::mate(Creature& lhs, Creature& rhs, seed_type seed)
{
  lhs.m_matingCooldown = MATING_COOLDOWN;
  rhs.m_matingCooldown = MATING_COOLDOWN;

  if(!lhs.takeEnergy(Creature::MAX_ENERGY * 0.2) || !rhs.takeEnergy(Creature::MAX_ENERGY * 0.2))
    return std::nullopt; // Mating without sufficient energy does nothing

  //TODO: implement inheritance
  return Creature(NeuralNetwork::cross(lhs.m_neuralNetwork, rhs.m_neuralNetwork, seed), (lhs.m_position+rhs.m_position)/2.0, 0.3 * MAX_ENERGY);
}

void Creature::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
  sf::CircleShape circleShape;
  {
    static constexpr float RADIUS_MULTIPIER = 10.0f;
    float radius = (m_health / MAX_HEALTH) * RADIUS_MULTIPIER;

    circleShape.setRadius(radius);
    circleShape.setOrigin({radius, radius});

    circleShape.setFillColor(lerp(sf::Color::Yellow, sf::Color::Green, m_energy / MAX_ENERGY));

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
