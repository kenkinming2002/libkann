#pragma once

#include "../Population.hpp"

#include <filesystem>

namespace App
{
  class GeneratePopulations
  {
  public:
    static constexpr size_t POPULATION_SIZE = 5000;
    inline static const auto HIDDEN_LAYERS = std::vector<size_t>{64, 64, 64, 64};

  public:
    static constexpr size_t SELECTION_ITERATIONS = 10;

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
