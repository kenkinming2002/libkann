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
      auto directoryPath = m_outputDirectory / (std::string("population")+std::to_string(i));
      m_population.writeTo(directoryPath);
      m_population.select();
    }
  }
}
