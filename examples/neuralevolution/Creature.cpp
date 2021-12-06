#include "Creature.hpp"

#include "World.hpp"
#include "Ray.hpp"
#include "libkann/RecurrentNeuralNetwork.hpp"

#include <libkann/WeightLayer.hpp>
#include <libkann/ActivationLayer.hpp>

#include <cassert>
#include <cmath>

kann::RecurrentNeuralNetwork Creature::makeNeuralNetork(const NeuralNetworkConfig& config, std::default_random_engine& engine)
{
  kann::RecurrentNeuralNetwork nn(config.memory);

  const auto activationFunction = kann::ActivationFunction(kann::ActivationFunction::Type::TANH);

  // TODO: do not create a vector for this
  std::vector<size_t> topology;
  topology.push_back(Creature::INPUT_COUNT + config.memory);
  topology.insert(topology.end(), config.hiddenLayers.begin(), config.hiddenLayers.end());
  topology.push_back(Creature::OUTPUT_COUNT + config.memory);

  for(size_t i=0; i < topology.size()-1; ++i)
  {
    size_t prevSize = topology[i];
    size_t nextSize = topology[i+1];
    auto weightLayer = std::make_unique<kann::WeightLayer>(prevSize, nextSize);

    auto activationLayer = std::make_unique<kann::ActivationLayer>(nextSize, activationFunction);
    nn.addLayer(std::move(weightLayer));
    nn.addLayer(std::move(activationLayer));
  }

  nn.randomize(engine);

  return nn;
}

static constexpr double ANGLE = M_PI / 12.0;

Creature::Creature(const Config& config, kann::RecurrentNeuralNetwork neuralNetwork,
    Eigen::Vector2d position, double energy, double health)
  : PhantomBody(position, config.maxRadius),
    m_config(config),
    m_neuralNetwork(std::move(neuralNetwork)),
    m_energy(energy), m_health(health),
    m_eyes{Eye(-ANGLE), Eye(ANGLE)} {}

void Creature::updateSight(World& world)
{
  Ray rays[EYES_COUNT];
  for(size_t i=0; i<EYES_COUNT; ++i)
  {
    auto& eye = m_eyes[i];
    auto& ray = rays[i];

    // MEMORIAL: The following line costs hours of debugging to add
    //
    // Thanks: gcc asan
    //
    // This following line of code has cost hours of debugging to add
    // This is because if eye.target is not set, (perhaps because no creature is
    // in front of us), the previous result will be used which may well be
    // freed. This manifest itself into errors in mating process.
    //
    //
    // Reset eye from last update
    eye.target = std::monostate{};
    eye.distance = m_config.viewDistance;
    ray = Ray(this->position(), this->direction() + eye.angle);
  }

  world.creatures().query(this->position(), m_config.viewDistance + m_config.maxRadius, [&](Creature& creature){
    for(size_t i=0; i<EYES_COUNT; ++i)
    {
      auto& eye = m_eyes[i];
      auto& ray = rays[i];
      if(&creature == this)
        return;

      if(double distance = ray.cast(creature); distance < eye.distance && distance > 0.0)
      {
        eye.distance = distance;
        eye.target = std::ref(creature);
      }
    }
  });
  world.berryBushes().query(this->position(), m_config.viewDistance + CONFIG.berryBush.radius, [&](BerryBush& berryBush){
    for(size_t i=0; i<EYES_COUNT; ++i)
    {
      auto& eye = m_eyes[i];
      auto& ray = rays[i];
      if(double distance = ray.cast(berryBush); distance < eye.distance && distance > 0.0)
      {
        eye.distance = distance;
        eye.target = std::ref(berryBush);
      }
    }
  });
}

void Creature::updateNeuralNetwork()
{
  // 1: Neural network
  Eigen::VectorXd input(INPUT_COUNT);
  input(INPUT_ENERGY)          = m_energy;
  input(INPUT_HEALTH)          = m_health;
  input(INPUT_VIEW_DISTANCE_0) = m_eyes[0].distance;
  input(INPUT_VIEW_DISTANCE_1) = m_eyes[1].distance;
  m_neuralNetwork.feedForward(std::move(input));
}

void Creature::updateCooldown(float dt)
{
  m_eatingCooldown -= dt;
  if(m_eatingCooldown<0.0f)
    m_eatingCooldown=0.0f;

  m_matingCooldown -= dt;
  if(m_matingCooldown<0.0f)
    m_matingCooldown=0.0f;
}

void Creature::updateSurvival(float dt)
{
  this->takeEnergy(m_config.passiveEnergyDrain * dt);
}

void Creature::updateStatistics(float dt)
{
  m_statistics.lifetime += dt;
}

void Creature::updateMovement(float dt, const World& world)
{
  const auto& output = m_neuralNetwork.output();

  auto linearSpeedFactor = output(OUTPUT_ACCELERATION_FACTOR);
  auto linearSpeedMultiplier = linearSpeedFactor >= 0.0 ?  m_config.forwardLinearSpeed : m_config.backwardLinearSpeed;
  auto linearSpeed = linearSpeedFactor * linearSpeedMultiplier;

  auto angularDirection = output(OUTPUT_RELATIVE_DIRECTION) * M_PI; // Map from (-1, 1) to (-PI, PI)

  this->applyImpulse(linearSpeed, angularDirection);
  this->takeEnergy(m_config.movementEnergyDrainMultiplier * linearSpeed * linearSpeed * dt);

  PhantomBody::update(dt, world);
}

void Creature::updateEating()
{
  static constexpr double EATING_ENERGY_COST = 5.0f;
  const double EATING_DISTANCE = m_config.maxRadius * 1.5 + CONFIG.berryBush.radius;

  const auto& output = m_neuralNetwork.output();
  if(output(OUTPUT_EATING_DESIRE)<0.0)
    return; // Don't want to eat

  if(m_eatingCooldown!=0.0)
    return; // Cooldown

  this->takeEnergy(EATING_ENERGY_COST);
  m_eatingCooldown = m_config.eatingCooldown;

  for(auto& eye: m_eyes)
  {
    if(!std::holds_alternative<std::reference_wrapper<BerryBush>>(eye.target))
      continue;

    auto& berryBush = std::get<std::reference_wrapper<BerryBush>>(eye.target).get();

    if(auto squaredDistance = (this->position() - berryBush.position()).squaredNorm();
        squaredDistance >= EATING_DISTANCE * EATING_DISTANCE)
      continue;

    if(berryBush.count() == 0)
      continue; // No berry to eat

    berryBush.take();
    m_energy += CONFIG.berryBush.energyPerBerry;
    if(m_energy>m_config.maxEnergy)
    {
      this->takeHealth(m_energy-m_config.maxEnergy);
      m_energy = m_config.maxEnergy;
    }

    return;
  }
}

void Creature::updateMating(World& world)
{
  static constexpr double MATING_ENERGY_COST = 5.0f;
  const double MATING_DISTANCE = m_config.maxRadius * 2.5;

  for(auto& eye: m_eyes)
  {
    if(!std::holds_alternative<std::reference_wrapper<Creature>>(eye.target))
      continue;

    auto& otherCreature = std::get<std::reference_wrapper<Creature>>(eye.target).get();

    const auto& output      = m_neuralNetwork.output();
    const auto& otherOutput = otherCreature.m_neuralNetwork.output();

    if(output(OUTPUT_MATING_DESIRE)<0.0 || otherOutput(OUTPUT_MATING_DESIRE)<0.0)
      return; // Don't want to mate

    if(m_matingCooldown!=0.0 || otherCreature.m_matingCooldown != 0.0)
      return; // Cooldown

    this->takeEnergy(MATING_ENERGY_COST);

    m_matingCooldown = m_config.matingCooldown;
    otherCreature.m_matingCooldown = m_config.matingCooldown;

    if(auto squaredDistance = (this->position() - otherCreature.position()).squaredNorm();
        squaredDistance >= MATING_DISTANCE * MATING_DISTANCE)
      continue; // Too far away

    if(!takeEnergy(m_config.maxEnergy * 0.2) || !otherCreature.takeEnergy(m_config.maxEnergy * 0.2))
      continue;

    auto neuralNetwork = m_neuralNetwork.cross(otherCreature.m_neuralNetwork, world.prng(), CONFIG.neuralNetwork.mutationRate);
    Eigen::Vector2d position = (this->position() + otherCreature.position()) / 2.0;

    auto newCreature = Creature(m_config, std::move(neuralNetwork),
      position,
      m_config.maxEnergy * 0.3,
      m_config.maxHealth
    );
    world.addCreature(std::move(newCreature));
    ++m_statistics.matingCount;
    ++otherCreature.m_statistics.matingCount;

    return;
  }
}

void Creature::updateHealth(float dt)
{
  // Take health
  if(m_energy == 0.0f)
    takeHealth(m_config.hungerHealthDrain * dt);

  // Healing
  if(m_energy >= m_config.maxEnergy * m_config.healingThreshold)
  {
    double amount = std::min(m_config.maxHealth - this->health(), m_energy);
    this->health(this->health()+amount);
    m_energy -= amount;
  }
}

bool Creature::takeEnergy(double amount)
{
  m_energy = std::max(m_energy-amount, 0.0);
  return m_energy != 0.0;
}

bool Creature::takeHealth(double amount)
{
  this->health(std::max(this->health() - amount, 0.0));
  return this->health() == 0.0;
}
