#include "GeneratePopulations.hpp"

#include <iostream>

namespace App
{
  GeneratePopulations::GeneratePopulations(const char* outputDirectory, seed_type seed) 
    : m_outputDirectory(outputDirectory), m_population(seed, POPULATION_SIZE, HIDDEN_LAYERS) 
  {
    std::clog << "Running with seed " << seed << '\n';
  }

  void GeneratePopulations::run()
  {
    // Initial
    auto write = [this](size_t i) {
      auto directory = m_outputDirectory / ("population" + std::to_string(i));
      m_population.writeTo(directory);
    };

    write(0);
    for(size_t i=1;;++i)
    {
      std::clog << "Generating population " << i << "...\n";
      m_population.select(SELECTION_ITERATIONS);
      write(i);
      std::clog << "Population " << i << " generated\n";
    }
  }
}
