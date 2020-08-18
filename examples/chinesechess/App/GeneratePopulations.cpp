#include "GeneratePopulations.hpp"

namespace App
{
  GeneratePopulations::GeneratePopulations(const char* outputDirectory, seed_type seed) 
    : m_outputDirectory(outputDirectory), m_population(seed, POPULATION_SIZE, HIDDEN_LAYERS) 
  {
    std::clog << "Running with seed " << seed << '\n';
  }

  void GeneratePopulations::run()
  {
    for(size_t i=0;;++i)
    {
      auto directory = m_outputDirectory / ("population" + std::to_string(i));
      m_population.writeTo(directory);
      std::cout << "Population " << i << " generated\n";
      m_population.select(SELECTION_ITERATIONS);
    }
  }
}
