#pragma once

#include "Config.hpp"
#include "BerryBush.hpp"
#include "Selectable.hpp"
#include "PhantomBody.hpp"

#include <libkann/RecurrentNeuralNetwork.hpp>

#include <Eigen/Eigen>

#include <functional>
#include <random>
#include <variant>

class World;

class Creature : public PhantomBody, public Selectable
{
private:
  static constexpr size_t EYES_COUNT = 2;

public:
  //static constexpr size_t NUM_INPUT = 4; // Energy, health, Sight
  enum Input {
    INPUT_ENERGY = 0,
    INPUT_HEALTH = 1,
    INPUT_VIEW_DISTANCE_0 = 2,
    INPUT_VIEW_DISTANCE_1 = 3,
    INPUT_COUNT
  };

  enum Output {
    OUTPUT_ACCELERATION_FACTOR = 0,
    OUTPUT_RELATIVE_DIRECTION = 1,
    OUTPUT_EATING_DESIRE = 2,
    OUTPUT_MATING_DESIRE = 3,
    OUTPUT_COUNT
  };


public:
  struct NeuralNetworkConfig
  {
    std::vector<size_t> hiddenLayers;
    size_t memory;
  };

  static kann::RecurrentNeuralNetwork makeNeuralNetork(const NeuralNetworkConfig& config, std::default_random_engine& engine);

  struct Config
  {
    // Maximum
    double maxRadius;
    double maxEnergy;
    double maxHealth;

    // Physics
    double forwardLinearSpeed;
    double backwardLinearSpeed;
    double angularSpeed;

    // Energy
    double passiveEnergyDrain;
    double movementEnergyDrainMultiplier;

    // Health
    double healingThreshold;
    double healingRate;

    double hungerHealthDrain;

    // Cooldown
    float eatingCooldown;
    float matingCooldown;

    // Intelligence
    double viewDistance;
  };
  /* Create a default creature from config using engine. You should use
   * setters to further configure the creature if so needed. */
  Creature(const Config& config, kann::RecurrentNeuralNetwork neuralNetwork,
      Eigen::Vector2d position, double energy, double health);

private:
  // TODO: Do not store it by value
  Config m_config;


public:
  template<typename InputIterator>
  static void batchUpdate(InputIterator first, InputIterator last, float dt, World& world);

private:
  void updateSight(World& world);
  void updateNeuralNetwork();
  void updateCooldown(float dt);
  void updateSurvival(float dt);
  void updateStatistics(float dt);
  void updateMovement(float dt, const World& world);
  void updateEating();
  void updateMating(World& world);
  void updateHealth(float dt);

private:
  bool takeEnergy(double amount);
  bool takeHealth(double amount);

public:
  bool dead() const { return m_health == 0.0; }
  bool healthy() const { return m_health == CONFIG.creature.maxHealth; }

public:
  double health() const { return m_health; }
  void health(double health) { m_health = health; this->radius() = CONFIG.creature.radius * m_health / CONFIG.creature.maxHealth; }

public:
  double energy() const { return m_energy; }
  void energy(double energy) { m_energy = energy; }

public:
  auto statistics() const { return m_statistics; }

public:
  friend class Renderer;

private:
  kann::RecurrentNeuralNetwork m_neuralNetwork;

private:
  double m_energy, m_health;

private:
  float m_eatingCooldown = 0.0f;
  float m_matingCooldown = 0.0f;

private:
  struct Eye
  {
  public:
    Eye(double angle) : angle(angle) {}

  public:
    double angle;
    double distance;

  public:
    std::variant<std::monostate, std::reference_wrapper<Creature>, std::reference_wrapper<BerryBush>> target;
  };
  Eye m_eyes[EYES_COUNT];

private:
  struct Statistics
  {
    float lifetime = 0.0f; //< How long has this creature survived
    size_t matingCount = 0;
  } m_statistics;
};

template<typename InputIterator>
void Creature::batchUpdate(InputIterator first, InputIterator last, float dt, World& world)
{
#pragma omp parallel
  {
#pragma omp for
    for(auto it = first; it != last; ++it)
    {
      Creature& creature = *it;
      creature.updateSight(world);
      creature.updateNeuralNetwork();
    }

    // NOTE: updateMovement() need to be called after all updateSight() and
    //       updateNeuralNetwork() complete, and there are 2 reasons.
    //
    //       1. updateSight() depends on the position of *ALL* creatures. However,
    //          updateMovement() updates the position of creatures.
    //
    //       2. updateMovement() depends on result from updateNeuralNetwork()
    //          which depends onn updateSight()
#pragma omp for
    for(auto it = first; it != last; ++it)
    {
      Creature& creature = *it;
      creature.updateCooldown(dt);
      creature.updateSurvival(dt);
      creature.updateStatistics(dt);
      creature.updateMovement(dt, world);
    }

#pragma omp single
    // Note: updateEating() and updateMating() must be ran in serial.
    for(auto it = first; it != last; ++it)
    {
      Creature& creature = *it;
      creature.updateEating();
      creature.updateMating(world);
    }

#pragma omp for
    for(auto it = first; it != last; ++it)
    {
      Creature& creature = *it;
      creature.updateHealth(dt);
    }
  }
}
