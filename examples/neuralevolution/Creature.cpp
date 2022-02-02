#include "Creature.hpp"

#include "World.hpp"
#include "Renderer.hpp"

#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>
#include <libkann/layers/RecurrentLayer.hpp>
#include <libkann/Model.hpp>

#include <cassert>
#include <cmath>

std::shared_ptr<kann::Model> Creature::makeNeuralNetork(const ModelConfig& config, std::default_random_engine& engine)
{
  /* Note: Model refer to Layer but has their own parameters and states To be
   *       able to cross different model, they have to have the same underlying
   *       layer. */
  static std::shared_ptr<kann::Layer> layer = [&config]()
  {

    // TODO: do not create a vector for this
    std::vector<size_t> topology;
    topology.push_back(Creature::INPUT_COUNT + config.memory);
    topology.insert(topology.end(), config.hiddenLayers.begin(), config.hiddenLayers.end());
    topology.push_back(Creature::OUTPUT_COUNT + config.memory);

    const auto activationFunction = kann::ActivationFunction(kann::ActivationFunction::Type::TANH);

    auto layer = std::make_shared<kann::RecurrentLayer>(config.memory);
    for(size_t i=0; i < topology.size()-1; ++i)
    {
      size_t prevSize = topology[i];
      size_t nextSize = topology[i+1];
      layer->addLayer(std::make_shared<kann::WeightLayer>(prevSize, nextSize));
      layer->addLayer(std::make_shared<kann::ActivationLayer>(nextSize, activationFunction));
    }
    return layer;
  }();

  auto model = std::make_shared<kann::Model>(layer);
  model->randomize();
  return model;
}

static constexpr double ANGLE = M_PI / 12.0;

Creature::Creature(b2World& world, const Config& config,
    std::shared_ptr<kann::Model> model, b2Vec2 position, double energy,
    double health)
  : Entity(Entity::Type::CREATURE, world, position, config.maxRadius),
    m_model(std::move(model)),
    m_eyes{Eye(-ANGLE), Eye(ANGLE)},
    m_energy(energy), m_health(health) {}

// All the thinking happen here
void Creature::updatePerception(const Config& config, float dt)
{
  // 1: Update sight
  for(auto& eye : m_eyes)
  {
    RaycastResult result;
    if(this->raycast(eye.angle, config.viewDistance, result))
    {
      eye.target   = result.entity;
      eye.distance = result.distance;
    }
    else
    {
      eye.target   = nullptr;
      eye.distance = config.viewDistance;
    }
  }
}

void Creature::updateModel(const Config& config)
{
  auto input = std::make_shared<kann::Tensor>(INPUT_COUNT);

  input->asArray()(INPUT_ENERGY)          = m_energy / config.maxEnergy;
  input->asArray()(INPUT_HEALTH)          = m_health / config.maxHealth;
  input->asArray()(INPUT_VIEW_DISTANCE_0) = m_eyes[0].distance / config.viewDistance;
  input->asArray()(INPUT_VIEW_DISTANCE_1) = m_eyes[1].distance / config.viewDistance;

  m_output = m_model->predict(std::move(input));
}

// Working alone
void Creature::update(const Config& config, const BerryBush::Config& berryBushConfig, float dt, World& world)
{
  static constexpr double EATING_ENERGY_COST = 5.0;
  static constexpr double MATING_ENERGY_COST = 5.0;
  static constexpr float REACH_MULTIPLIER = 1.5f;

  // 1: Cooldown
  {
    m_eatingCooldown -= dt;
    if(m_eatingCooldown<0.0f)
      m_eatingCooldown=0.0f;

    m_matingCooldown -= dt;
    if(m_matingCooldown<0.0f)
      m_matingCooldown=0.0f;
  }

  // 2: Survival energy cost
  {
    this->takeEnergy(config.passiveEnergyDrain * dt);
  }

  // 3: Movement
  {
    // TODO: Fix the scale of the world, we do not want 1 pixel per meter
    const float TEMPORARY_HACK = 1000.0f;

    // TODO: Rename configuration variable to suit the changes in their meaning
    auto linearForceFactor = m_output->asArray()(OUTPUT_LINEAR_FORCE_FACTOR);
    auto linearForceMultiplier = linearForceFactor >= 0.0 ?  config.forwardLinearSpeed : config.backwardLinearSpeed;
    auto linearForce = linearForceFactor * linearForceMultiplier * dt;
    this->applyForwardForce(linearForce * TEMPORARY_HACK);

    // TODO: Add config variable for angular force or not?
    auto angularForceFactor = m_output->asArray()(OUTPUT_ANGULAR_FORCE_FACTOR); // Map from (-1, 1) to (-PI, PI)
    auto angularForce =  angularForceFactor * M_PI * dt;
    this->applyTorque(angularForce);
    this->takeEnergy(config.movementEnergyDrainMultiplier * linearForce * linearForce * dt);

    // 4: Hunger
    if(m_energy == 0.0f)
      takeHealth(config.hungerHealthDrain * dt);

    // 5: Healing
    if(m_energy >= config.maxEnergy * config.healingThreshold)
    {
      double amount = std::min(config.maxHealth - this->health(), m_energy);
      this->health(this->health()+amount);
      m_energy -= amount;
    }
  }

  // 4: Eating
  {
    if(m_output->asArray()(OUTPUT_EATING_DESIRE)>0.0 && m_eatingCooldown == 0.0)
    {
      takeEnergy(EATING_ENERGY_COST);
      m_eatingCooldown = config.eatingCooldown;
      for(auto& eye: m_eyes)
      {
        auto* target = eye.target;
        if(!target)
          continue;

        if(target->type() != Entity::Type::BERRY_BUSH)
          continue;

        BerryBush* berryBush = static_cast<BerryBush*>(target);

        const auto eatingDistance = (this->radius() + berryBush->radius()) * REACH_MULTIPLIER;
        if(eye.distance > eatingDistance)
          continue; // Too far away

        if(berryBush->count() == 0)
          continue; // No berry to eat

        m_energy += berryBush->take(berryBushConfig);

        // Clamping
        if(m_energy>config.maxEnergy)
        {
          this->takeHealth(m_energy-config.maxEnergy); // Penalty for eating too much
          m_energy = config.maxEnergy;
        }

        break; // Only eat from one berry bush at once
      }
    }
  }

  // 5: Mating
  {
    if(m_output->asArray()(OUTPUT_MATING_DESIRE)>0.0 && m_matingCooldown == 0.0)
    {
      takeEnergy(MATING_ENERGY_COST);
      m_matingCooldown = config.matingCooldown;
      for(auto& eye: m_eyes)
      {
        auto* target = eye.target;
        if(!target)
          continue;

        if(target->type() != Entity::Type::CREATURE)
          continue;

        Creature* other = static_cast<Creature*>(target);

        const auto matingDistance = (radius() + other->radius()) * REACH_MULTIPLIER;
        if(eye.distance > matingDistance)
          continue;

        // Check if other have the same desire
        if(other->m_output->asArray()(OUTPUT_EATING_DESIRE)<0.0)
          continue; // Other do not want to mate

        if(other->m_matingCooldown != 0.0)
          continue; // Other cannot mate

        // Yes, he/she do, perhaps the cost would be different?
        other->takeEnergy(MATING_ENERGY_COST);
        other->m_matingCooldown = config.matingCooldown;


        if(!takeEnergy(config.maxEnergy * 0.2) || !other->takeEnergy(config.maxEnergy * 0.2))
          continue;

        auto newModel = kann::cross(*m_model, *other->m_model, world.prng(), config.mutationRate);
        auto newPosition = position();
        newPosition += other->position();
        newPosition *= 0.5;

        // What we really need is the world prng and b2World
        auto newCreature = Creature(world.world(), config,
            std::move(newModel), newPosition,
            config.maxEnergy * 0.3, config.maxHealth
        );
        world.addCreature(std::move(newCreature));

        ++m_statistics.matingCount;
        ++other->m_statistics.matingCount;
        break; // Only mate once
      }
    }
  }

  // 6: Statistics
  {
    m_statistics.lifetime += dt;
  }
}

void Creature::draw(const Config& config, Renderer& renderer) const
{
  const auto radius = this->radius();
  const auto color = lerp(sf::Color::Yellow, sf::Color::Green, m_energy / config.maxEnergy);
  const auto position = this->position();
  renderer.addCircle({position.x, position.y}, radius, color);

  if(renderer.debug())
  {
    for(const auto& eye: m_eyes)
    {
      if(eye.distance == config.viewDistance)
        continue;

      float angle = this->angle() + eye.angle;
      renderer.addLine({position.x, position.y}, eye.distance, angle, 3.0f, sf::Color::Red);
    }
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

