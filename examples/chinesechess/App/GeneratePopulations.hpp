#pragma once

#include "../Population.hpp"

#include <filesystem>

namespace App
{
  class GeneratePopulations
  {
  public:
    static void usage();
    static int main(int argc, char* argv[]);

  public:
    using seed_type = Population::seed_type;
    GeneratePopulations(const char* outputDirectory, seed_type populationSeed, size_t populationSize, size_t selectionIterationsCount);

  public:
    void run();

  private:
    std::filesystem::path m_outputDirectory;
    Population m_population;
    size_t m_selectionIterations;
  };
}
