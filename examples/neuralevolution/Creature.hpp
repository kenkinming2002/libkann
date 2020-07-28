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
  Creature(NeuralNetwork neuralNetwork, Eigen::Vector2d position, double energy = CONFIG.creature.maxEnergy, 
      double health = CONFIG.creature.maxHealth);
  template<typename PRNG>
  Creature(PRNG& prng, Eigen::Vector2d position);

public:
  /*
   * MT-Safe
   *
   * Even though a non-const reference to world is passed, it is not allowed to
   * modify content of world, but only store reference to variable in world for
   * later modification.
   *
   * This updates attributes of creatures that does not requires neurological
   * thinking and also feed the neural brain.
   */
  void preUpdate(float dt, World& world);

  /*
   * MT-Unsafe
   *
   * Mostly not rhread safe. Some thread safe component can probaly be mvoed to
   * postUpdate but the gain is questionable.
   */
  void update(float dt, World& world);

public:
  /*
   * The following functions are called in preUpdate
   */
  void updateSight(World& world);
  void updateCooldown(float dt);
  void updateSurvival(float dt);
  void updateStatistics(float dt);

private:
  /*
   * The following functions are called in update
   */
  void updateMovement(float dt, World& world);
  void updateEating();
  void updateMating(World& world);

private:
  inline bool takeEnergy(double amount);
  inline bool takeHealth(double amount);

public:
  bool dead() const { return m_health == 0.0; }
  bool healthy() const { return m_health == CONFIG.creature.maxHealth; }

public:
  double health() const { return m_health; }
  void health(double health) 
  { 
    m_health = health; 
    this->radius() = CONFIG.creature.radius * m_health / CONFIG.creature.maxHealth;
  }

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
