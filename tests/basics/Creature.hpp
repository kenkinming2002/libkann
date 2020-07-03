#pragma once

#include <Eigen/Eigen>
#include <libkann/NeuralNetwork.hpp>

#include <SFML/Graphics/Drawable.hpp>
#include <ostream>

#include <utility>
#include <optional>

#include "BerryBush.hpp"

class World;

class Creature : public sf::Drawable
{
private:
  static constexpr size_t NUM_INPUT = 6; // Energy, health, Offset of closest berry bushes and closest creature
  static constexpr size_t NUM_OUTPUT = 4; // Linear, angular speed, eating and mating desire

public:
  template<typename PRNG>
  Creature(PRNG& prng, Eigen::Vector2d position);
  Creature(NeuralNetwork neuralNetwork, Eigen::Vector2d position, double energy);

public:
  void update(float dt, World& world);

public:
  void updateSight(World& world);
  void updateEating();
  void updateMating(World& world);

public:
  bool dead() const { return m_health == 0.0; }

private:
  bool takeEnergy(double amount);
  bool takeHealth(double amount);

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
    std::reference_wrapper<BerryBush> closestBerryBush;
    std::reference_wrapper<Creature>  closestCreature;
  };
  std::optional<Sight> m_sight;
};
