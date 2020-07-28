#pragma once

#include "Config.hpp"
#include "BerryBush.hpp"
#include "Selectable.hpp"
#include "PhantomBody.hpp"

#include <libkann/NeuralNetwork.hpp>

#include <Eigen/Eigen>

#include <functional>
#include <variant>

class World;

class Creature : public PhantomBody, public Selectable
{
private:
  static constexpr size_t EYES_COUNT = 2;

private:
  static constexpr size_t NUM_INPUT = 4; // Energy, health, Sight
  enum class Output {
    ACCELERATION_FACTOR = 0,
    RELATIVE_DIRECTION = 1,
    EATING_DESIRE = 2,
    MATING_DESIRE = 3,
    COUNT
  };

private:
  static const std::vector<size_t>& topology();

public:
  Creature(NeuralNetwork neuralNetwork, Eigen::Vector2d position, double energy = CONFIG.creature.maxEnergy, double health = CONFIG.creature.maxHealth);
  template<typename PRNG>
  Creature(PRNG& prng, Eigen::Vector2d position);

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

  auto statistics() const { return m_statistics; }

public:
  friend class Renderer;

private:
  NeuralNetwork m_neuralNetwork;

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
