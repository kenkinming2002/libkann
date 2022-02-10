#pragma once

#include "Agent.hpp"

#include <random>
#include <vector>

#include <filesystem>

class Population
{
public:
  using random_engine_type = std::default_random_engine;
  using seed_type = random_engine_type::result_type;

public:
  Population(seed_type seed, size_t size);

public:
  void select(size_t iterations, double mutationRate = 0.05);

public:
  void writeTo(const std::filesystem::path& directory) const;

private:
  std::vector<Agent> m_agents;

private:
  random_engine_type m_generator;
};
