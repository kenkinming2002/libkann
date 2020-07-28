#include "Creature.hpp"

#include "World.hpp"
#include "Ray.hpp"

#include <cassert>
#include <cmath>

static constexpr double ANGLE = M_PI / 12.0;

const std::vector<size_t>& Creature::topology()
{
  static std::vector<size_t> topology = [](){
    std::vector<size_t> topology;

    topology.reserve(CONFIG.creature.hiddenLayers.size() + 2);

    topology.push_back(Creature::NUM_INPUT);
    topology.insert(topology.end(), CONFIG.creature.hiddenLayers.begin(), CONFIG.creature.hiddenLayers.end());
    topology.push_back(static_cast<size_t>(Creature::Output::COUNT));

    return topology;
  }();
  return topology;
}

Creature::Creature(NeuralNetwork neuralNetwork, Eigen::Vector2d position, double energy, double health) 
  : PhantomBody(position, CONFIG.creature.radius), m_neuralNetwork(neuralNetwork),
    m_energy(energy), m_health(health), m_eyes{Eye(-ANGLE), Eye(ANGLE)} {}

template<typename PRNG>
Creature::Creature(PRNG& prng, Eigen::Vector2d position) 
  : Creature(NeuralNetwork(topology(), CONFIG.creature.memory, prng), position) {}

template Creature::Creature(std::mt19937& prng, Eigen::Vector2d position);

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
    eye.distance = CONFIG.creature.viewDistance;
    ray = Ray(this->position(), this->direction() + eye.angle);
  }

  world.creatures().query(this->position(), CONFIG.creature.viewDistance + CONFIG.creature.radius, [&](Creature& creature){
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
  world.berryBushes().query(this->position(), CONFIG.creature.viewDistance + CONFIG.berryBush.radius, [&](BerryBush& berryBush){
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
  m_neuralNetwork.input({m_energy, this->health(), m_eyes[0].distance, m_eyes[1].distance});
  m_neuralNetwork.feedForward();
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
  this->takeEnergy(CONFIG.creature.passiveEnergyDrain * dt);
}

void Creature::updateStatistics(float dt)
{
  m_statistics.lifetime += dt;
}

void Creature::updateMovement(float dt, const World& world)
{
  auto linearSpeedFactor = m_neuralNetwork.output(Output::ACCELERATION_FACTOR);
  auto linearSpeedMultiplier = linearSpeedFactor >= 0.0 ?  CONFIG.creature.forwardLinearSpeed : CONFIG.creature.backwardLinearSpeed;
  auto linearSpeed = linearSpeedFactor * linearSpeedMultiplier;

  auto angularDirection = m_neuralNetwork.output(Output::RELATIVE_DIRECTION) * M_PI; // Map from (-1, 1) to (-PI, PI)

  this->applyImpulse(linearSpeed, angularDirection);
  this->takeEnergy(CONFIG.creature.movementEnergyDrainMultiplier * linearSpeed * linearSpeed * dt);

  PhantomBody::update(dt, world);
}

void Creature::updateEating()
{
  static constexpr double EATING_ENERGY_COST = 5.0f;
  const double EATING_DISTANCE = CONFIG.creature.radius * 1.5 + CONFIG.berryBush.radius;

  if(m_neuralNetwork.output(Output::EATING_DESIRE)<0.0)
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

    if(auto squaredDistance = (this->position() - berryBush.position()).squaredNorm();
        squaredDistance >= EATING_DISTANCE * EATING_DISTANCE)
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

    if(m_neuralNetwork.output(Output::MATING_DESIRE)<0.0 || otherCreature.m_neuralNetwork.output(Output::MATING_DESIRE)<0.0)
      return; // Don't want to mate

    if(m_matingCooldown!=0.0 || otherCreature.m_matingCooldown != 0.0)
      return; // Cooldown

    this->takeEnergy(MATING_ENERGY_COST);

    m_matingCooldown = CONFIG.creature.matingCooldown;
    otherCreature.m_matingCooldown = CONFIG.creature.matingCooldown;

    if(auto squaredDistance = (this->position() - otherCreature.position()).squaredNorm();
        squaredDistance >= MATING_DISTANCE * MATING_DISTANCE)
      continue; // Too far away

    if(!takeEnergy(CONFIG.creature.maxEnergy * 0.2) || !otherCreature.takeEnergy(CONFIG.creature.maxEnergy * 0.2))
      continue;

    auto neuralNetwork = NeuralNetwork::cross(m_neuralNetwork, otherCreature.m_neuralNetwork, world.prng(), CONFIG.neuralNetwork.mutationRate);
    Eigen::Vector2d position = (this->position() + otherCreature.position()) / 2.0;

    world.addCreature(Creature(std::move(neuralNetwork), position, 0.3 * CONFIG.creature.maxEnergy));
    ++m_statistics.matingCount;
    ++otherCreature.m_statistics.matingCount;

    return;
  }
}

void Creature::updateHealth(float dt)
{
  // Take health
  if(m_energy == 0.0f)
    takeHealth(CONFIG.creature.hungerHealthDrain * dt);

  // Healing
  if(m_energy >= CONFIG.creature.maxEnergy * CONFIG.creature.healingThreshold)
  {
    double amount = std::min(CONFIG.creature.maxHealth - this->health(), m_energy);
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
