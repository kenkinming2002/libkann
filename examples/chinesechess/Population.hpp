#pragma once

#include "Agent.hpp"

#include <random>
#include <vector>

#include <filesystem>

class Population
{
public:
  using random_engine_type = std::mt19937;
  using seed_type = random_engine_type::result_type;

public:
  Population(seed_type seed, size_t size, const std::vector<size_t>& agentsHiddenLayers);

public:
  void select(size_t iterations, double mutationRate = 0.05);

public:
  void writeTo(const std::filesystem::path& directory) const;

private:
  std::vector<Agent> m_agents;

private:
  std::mt19937 m_generator;
};
