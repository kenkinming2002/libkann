#pragma once

#include "BerryBush.hpp"
#include "Renderer.hpp"

#include <libkann/Model.hpp>

#include <Eigen/Eigen>

#include <functional>
#include <random>
#include <variant>

class World;

class Creature : public Entity
{
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
    OUTPUT_LINEAR_FORCE_FACTOR  = 0,
    OUTPUT_ANGULAR_FORCE_FACTOR = 1,
    OUTPUT_EATING_DESIRE = 2,
    OUTPUT_MATING_DESIRE = 3,
    OUTPUT_COUNT
  };


public:
  struct ModelConfig
  {
    std::vector<size_t> hiddenLayers;
    size_t memory;
  };

  static std::shared_ptr<kann::Model> makeNeuralNetork(const ModelConfig& config, std::default_random_engine& engine);

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

    // Mating
    double mutationRate;
  };

  /* Create a default creature from config using engine. You should use
   * setters to further configure the creature if so needed. */
  Creature(b2World& world, const Config& config, std::shared_ptr<kann::Model>
      model, b2Vec2 position, double energy, double health);

public:
  /* Not thread safe */
  void updatePerception(const Config& config, float dt);

  /* Thread safe - and really need to be parallelize because this is *SLOW* */
  void updateModel(const Config& config);

  /* Not thread safe */
  void update(const Config& config, const BerryBush::Config& berryBushConfig, float dt, World& world);

public:
  void draw(const Config& config, Renderer& renderer) const;

public:
  bool takeEnergy(double amount);
  bool takeHealth(double amount);

public:
  bool dead() const { return m_health == 0.0; }
  bool healthy(const Config& config) const { return m_health == config.maxHealth; }

public:
  double health() const { return m_health; }
  void health(double health) { m_health = health; }

public:
  double energy() const { return m_energy; }
  void energy(double energy) { m_energy = energy; }

public:
  auto statistics() const { return m_statistics; }

private:
  std::shared_ptr<kann::Model> m_model;
  std::shared_ptr<const kann::Tensor> m_output;

private:
  static constexpr size_t EYES_COUNT = 2;
  struct Eye
  {
  public:
    Eye(double angle) : angle(angle) {}

  public:
    double angle;

  public:
    Entity* target;
    double distance;
  };
  Eye m_eyes[EYES_COUNT];

private:
  double m_energy, m_health;

private:
  float m_eatingCooldown = 0.0f;
  float m_matingCooldown = 0.0f;

private:
  struct Statistics
  {
    float lifetime = 0.0f; //< How long has this creature survived
    size_t matingCount = 0;
  } m_statistics;
};

