#include "App.hpp"

App::App(std::filesystem::path outputDirectory, seed_type seed)
  : m_outputDirectory(std::move(outputDirectory)), m_population(seed, 100) {}

void App::run()
{
  for(size_t i=0;;++i)
  {
    auto directoryPath = m_outputDirectory / (std::string("population")+std::to_string(i));
    m_population.writeTo(directoryPath);
    m_population.select();
  }
}
