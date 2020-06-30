#pragma once

#include <stddef.h>
#include <optional>

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

    float eatingCooldown;
    float matingCooldown;
  };
  Creature creature;

  struct World  
  {
    double width;
    double height;
    size_t initialCreaturesCount;
    size_t initialBerryBushesCount;
  };
  World world;
} CONFIG;
