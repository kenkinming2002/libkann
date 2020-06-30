#pragma once

#include <Eigen/Eigen>
#include <libkann/NeuralNetwork.hpp>

#include <SFML/Graphics/Drawable.hpp>
#include <ostream>

#include <optional>

#include "BerryBush.hpp"

class World;

class Creature : public sf::Drawable
{
private:
  static constexpr size_t NUM_INPUT = 6; // Energy, health, Offset of closest berry bushes and closest creature
  static constexpr size_t NUM_OUTPUT = 4; // Linear, angular speed, eating and mating desire

public:
  using seed_type = typename NeuralNetwork::seed_type;
  using random_engine_type = typename NeuralNetwork::random_engine_type;

public:
  Creature(seed_type seed, Eigen::Vector2d position);
  Creature(NeuralNetwork neuralNetwork, Eigen::Vector2d position, double energy);

public:
  void updateSight(const World& world);
  void update(float dt, Eigen::Vector2d worldDimension);
  bool dead() const { return m_health == 0.0; }

private:
  bool takeEnergy(double amount);
  bool takeHealth(double amount);

public:
  bool canEat(const BerryBush& berryBush) const;
  void eat(BerryBush& berryBush);

public:
  static bool canMate(const Creature& lhs, const Creature& rhs);
  static std::optional<Creature> mate(Creature& lhs, Creature& rhs, seed_type seed);

public:
  double health() const { return m_health; }

public:
  static double distance(const Creature& lhs, const Creature& rhs) { return (lhs.m_position - rhs.m_position).norm(); }
  double distance(const BerryBush& berryBush) const { return (m_position - berryBush.position()).norm(); }

public:
  void draw(sf::RenderTarget &target, sf::RenderStates states) const override;
  friend std::ostream& operator<<(std::ostream& os, const Creature& creature);

private:
  NeuralNetwork m_neuralNetwork;

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
  struct Sight
  {
    const BerryBush* closestBerryBush = nullptr;

    Eigen::Vector2d closestBerryBushLocation = {0.0, 0.0};
    Eigen::Vector2d closestCreatureLocation = {0.0, 0.0};
  } m_sight;
};
