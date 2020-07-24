#pragma once

#include "Config.hpp"

#include "BerryBush.hpp"
#include "Selectable.hpp"

#include <Eigen/Eigen>
#include <libkann/NeuralNetwork.hpp>

#include <ostream>

#include <utility>
#include <optional>
#include <variant>

class World;

class Creature : public Selectable
{
private:
  static constexpr size_t EYES_COUNT = 2;

private:
  static constexpr size_t NUM_INPUT = 4; // Energy, health, Sight
  static constexpr size_t NUM_OUTPUT = 4; // Linear, angular speed, eating and mating desire

  enum class Output {
    LINEAR_SPEED_FACTOR = 0,
    ANGULAR_SPEED_FACTOR = 1,
    EATING_DESIRE = 2,
    MATING_DESIRE = 3
  };

private:
  static const std::vector<size_t>& topology();

public:
  template<typename PRNG>
  Creature(PRNG& prng, Eigen::Vector2d position);
  Creature(NeuralNetwork neuralNetwork, Eigen::Vector2d position, double energy);

public:
  void preUpdate(float dt, World& world) const;
  // Modify externally visible data
  void update(float dt, World& world);

public:
  void updateSight(World& world) const;

private:
  double updateMovement(float dt, World& world);
  double updateSurvival(float dt);

  void updateCooldown(float dt);
  void updateEating();
  void updateMating(World& world);

public:
  bool dead() const { return m_health == 0.0; }
  bool healthy() const { return m_health == CONFIG.creature.maxHealth; }

private:
  bool takeEnergy(double amount);
  bool takeHealth(double amount);

public:
  double health() const { return m_health; }
  auto position() const { return m_position; }

public:
  friend class Renderer;

private:
  mutable NeuralNetwork m_neuralNetwork;

private:
  Eigen::Vector2d m_position;
  double m_rotation;

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
  mutable Eye m_eyes[EYES_COUNT];
};
