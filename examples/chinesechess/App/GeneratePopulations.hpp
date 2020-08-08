#pragma once

#include "../Population.hpp"

#include <filesystem>

namespace App
{
  class GeneratePopulations
  {
  public:
    using random_engine_type = Population::random_engine_type;
    using seed_type = Population::seed_type;

  public:
    GeneratePopulations(const char* outputDirectory, seed_type seed);

  public:
    void run();

  private:
    std::filesystem::path m_outputDirectory;
    Population m_population;
  };
}
