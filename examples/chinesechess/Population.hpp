#pragma once

#include "Agent.hpp"

#include <random>
#include <vector>

class Population
{
public:
  using random_engine_type = std::mt19937;
  using seed_type = random_engine_type::result_type;

public:
  Population(seed_type seed, size_t size);

public:
  void select();

private:
  std::vector<Agent> m_agents;

private:
  std::mt19937 m_generator;
};
