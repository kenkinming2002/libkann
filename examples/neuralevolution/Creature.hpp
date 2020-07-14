#pragma once

#include "Config.hpp"

#include <Eigen/Eigen>
#include <libkann/NeuralNetwork.hpp>

#include <ostream>

#include <utility>
#include <optional>
#include <variant>

#include "BerryBush.hpp"

class World;

class Creature
{
private:
  static constexpr size_t EYES_COUNT = 2;

private:
  static constexpr size_t NUM_INPUT = 4; // Energy, health, Sight
  static constexpr size_t NUM_OUTPUT = 4; // Linear, angular speed, eating and mating desire

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
  static double distance(const Creature& lhs, const Creature& rhs) { return (lhs.m_position - rhs.m_position).norm(); }
  double distance(const BerryBush& berryBush) const { return (m_position - berryBush.position()).norm(); }

public:
  friend std::ostream& operator<<(std::ostream& os, const Creature& creature);

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
  double m_eatingDesire;
  double m_matingDesire;

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
