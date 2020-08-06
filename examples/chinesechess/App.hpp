#pragma once

#include <filesystem>
#include "Population.hpp"

class App
{
public:
  using random_engine_type = Population::random_engine_type;
  using seed_type = Population::seed_type;

public:
  App(std::filesystem::path outputDirectory, seed_type seed);

public:
  void run();

private:
  std::filesystem::path m_outputDirectory;
  Population m_population;
};
