#pragma once

#include <stddef.h>
#include <vector>

extern struct Config
{
public:
  Config() = default;
  Config(const char* filename);

public:
  struct NeuralNetwork
  {
    double mutationRate;
  };
  NeuralNetwork neuralNetwork;

  struct BerryBush
  {
    double energyPerBerry;
    size_t maxBerryCount;
    float growthRate;

    double radius;
  };
  BerryBush berryBush;

  struct Creature
  {
    double maxEnergy;
    double maxHealth;

    double forwardLinearSpeed;
    double backwardLinearSpeed;
    double angularSpeed;

    double passiveEnergyDrain;
    double movementEnergyDrainMultiplier;

    double hungerHealthDrain;

    double healingThreshold;
    double healingRate;

    float eatingCooldown;
    float matingCooldown;

    double radius;

    std::vector<size_t> hiddenLayers;
    size_t memory;

    double viewDistance;
  };
  Creature creature;

  struct World
  {
    float width;
    float height;
    size_t initialCreaturesCount;
    size_t initialBerryBushesClusterSizeMin;
    size_t initialBerryBushesClusterSizeMax;
    size_t initialBerryBushesClusterCount;
  };
  World world;
} CONFIG;
