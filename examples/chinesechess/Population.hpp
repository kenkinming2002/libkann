#pragma once

#include "AIAgent.hpp"

#include <filesystem>
#include <vector>

class Population
{
public:
  Population(size_t size);

public:
  void select(size_t iterations, std::default_random_engine& engine, double mutationRate);
  void write(const std::filesystem::path& path);

private:
  std::vector<AIAgent> m_agents;
};
