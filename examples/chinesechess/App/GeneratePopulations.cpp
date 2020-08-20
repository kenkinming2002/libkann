#include "GeneratePopulations.hpp"

#include <iostream>

namespace App
{
  GeneratePopulations::GeneratePopulations(const char* outputDirectory, seed_type populationSeed, size_t populationSize, const std::vector<size_t>& agentsHiddenLayers, size_t selectionIterationsCount) 
    : m_outputDirectory(outputDirectory), m_population(populationSeed, populationSize, agentsHiddenLayers), m_selectionIterations(selectionIterationsCount)
  {
    std::clog << "Running with seed " << populationSeed << '\n';
  }

  void GeneratePopulations::run()
  {
    // Initial
    auto write = [this](size_t i) {
      std::cout << "Writing population " << i << "..." << std::flush;
      m_population.writeTo(m_outputDirectory / ("population" + std::to_string(i)));
      std::cout << "Done\n";
    };

    write(0);
    for(size_t i=1;;++i)
    {
      std::clog << "Generating population " << i << "...\n";
      m_population.select(m_selectionIterations);
      write(i);
      std::clog << "Population " << i << " generated\n";
    }
  }
}
