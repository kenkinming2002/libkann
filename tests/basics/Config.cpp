#include "Config.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

Config::Config(const char* fileName)
{
  std::ifstream configFile(fileName);
  if(!configFile.is_open())
    throw std::runtime_error(std::string("Failed to open config file: ") + fileName);

  using namespace nlohmann;

  json config;
  configFile >> config;

  this->neuralNetwork.mutationRate = config.at("NeuralNetwork").at("MutationRate").get<double>();

  auto berryBushConfig = config.at("BerryBush");
  this->berryBush.energyPerBerry = berryBushConfig.at("EnergyPerBerry").get<double>();
  this->berryBush.maxBerryCount  = berryBushConfig.at("MaxBerryCount").get<size_t>();
  this->berryBush.growthRate     = berryBushConfig.at("GrowthRate").get<float>();

  this->berryBush.radius = berryBushConfig.at("Radius").get<double>();

  auto creatureConfig = config.at("Creature");
  this->creature.maxEnergy = creatureConfig.at("MaxEnergy").get<double>();
  this->creature.maxHealth = creatureConfig.at("MaxHealth").get<double>();

  // FIXME: use 2 variable in config file
  this->creature.forwardLinearSpeed = creatureConfig.at("LinearSpeed").get<double>();
  this->creature.backwardLinearSpeed = creatureConfig.at("LinearSpeed").get<double>();

  this->creature.angularSpeed = creatureConfig.at("AngularSpeed").get<double>();
  this->creature.passiveEnergyDrain = creatureConfig.at("PassiveEnergyDrain").get<double>();
  this->creature.movementEnergyDrainMultiplier = creatureConfig.at("MovementEnergyDrainMultiplier").get<double>();
  this->creature.hungerHealthDrain = creatureConfig.at("HungerHealthDrain").get<double>();

  this->creature.healingThreshold = creatureConfig.at("HealingThreshold").get<double>();
  this->creature.healingRate = creatureConfig.at("HealingRate").get<double>();

  this->creature.eatingCooldown = creatureConfig.at("EatingCooldown").get<float>();
  this->creature.matingCooldown = creatureConfig.at("MatingCooldown").get<float>();

  this->creature.radius = creatureConfig.at("Radius").get<double>();

  this->creature.memory = creatureConfig.at("Memory").get<size_t>();
  
  this->creature.viewDistance = creatureConfig.at("ViewDistance").get<double>();

  auto worldConfig = config.at("World");
  this->world.width = worldConfig.at("Width").get<double>();
  this->world.height = worldConfig.at("Height").get<double>();
  this->world.initialCreaturesCount = worldConfig.at("InitialCreaturesCount").get<size_t>();
  this->world.initialBerryBushesCount = worldConfig.at("InitialBerryBushesCount").get<size_t>();
}

Config CONFIG;
