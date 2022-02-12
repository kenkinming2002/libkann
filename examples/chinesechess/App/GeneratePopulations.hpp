#pragma once

#include "../Population.hpp"

#include <filesystem>
#include <random>

namespace App
{
  class GeneratePopulations
  {
  public:
    static void usage();
    static int main(int argc, char* argv[]);

  public:
    using seed_type = std::default_random_engine::result_type;
    GeneratePopulations(const char* outputDirectory, seed_type populationSeed, size_t populationSize, size_t selectionIterationsCount);

  public:
    void run();

  private:
    std::filesystem::path m_outputDirectory;

    Population m_population;

    std::default_random_engine m_engine;
    size_t m_selectionIterations;
  };
}
