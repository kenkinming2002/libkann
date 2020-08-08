#include "GeneratePopulations.hpp"

namespace App
{
  GeneratePopulations::GeneratePopulations(const char* outputDirectory, seed_type seed) 
    : m_outputDirectory(outputDirectory), m_population(seed, 100) 
  {
    std::clog << "Running with seed " << seed << '\n';
  }

  void GeneratePopulations::run()
  {
    for(size_t i=0;;++i)
    {
      auto directoryName = "population" + std::to_string(i);
      auto directory = m_outputDirectory / directoryName;
      m_population.writeTo(directory);
      m_population.select();
    }
  }
}
